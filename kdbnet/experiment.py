import copy
import numpy as np
import uncertainty_toolbox as uct
import learn2learn as l2l
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from kdbnet.dta import KIBA, DAVIS
from kdbnet.model import DTAModel
from kdbnet.metrics import evaluation_metrics
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
from kdbnet.utils import(
    Logger,
    Saver,
    EarlyStopping
)
torch.set_num_threads(1)

def _parallel_train_per_epoch(
    kwargs=None, test_loader=None,
    n_epochs=None, eval_freq=None, test_freq=None,
    monitoring_score='pearson',
    loss_fn=None, logger=None,
    test_after_train=True,
):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    stopper = kwargs['stopper']
    best_model_state_dict = kwargs['best_model_state_dict']
    if stopper.early_stop:
        return kwargs
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / step
        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )
        if test_freq is not None and epoch % test_freq == 0:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                )
        if stopper.early_stop:
            logger.info('Eearly stop at epoch {}'.format(epoch))
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    rets = dict(midx = midx, model = model)
    return rets

def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results

def _unpack_evidential_output(output):
    mu, v, alpha, beta = torch.split(output, output.shape[1]//4, dim=1)
    inverse_evidence = 1. / ((alpha - 1) * v)
    var = beta * inverse_evidence
    return mu, var, inverse_evidence

class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method='protein',
        split_frac=[0.7, 0.1, 0.2],
        prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=1, n_epochs=5, batch_size=256,
        lr=0.001,        
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        output_dir='../output', save_log=False
    ):
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)

        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        dataset_klass = {
            'kiba': KIBA,
            'davis': DAVIS,
        }[task]
        self.dataset = dataset_klass(
            split_method=split_method,
            split_frac=split_frac,
            seed=seed,
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self._task_data_df_split = None
        self._task_loader = None
        n_gpus = torch.cuda.device_count()
        if self.parallel and n_gpus < self.n_ensembles:
            self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
            f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
            )
        self.devices = [torch.device(f'cuda:{i % n_gpus}')
            for i in range(self.n_ensembles)]
        self.model_config = dict(
            prot_emb_dim=1280,
            prot_gcn_dims=prot_gcn_dims,
            prot_fc_dims=prot_fc_dims,
            drug_node_in_dim=[66, 1], 
            drug_node_h_dims=drug_gcn_dims,
            drug_fc_dims=drug_fc_dims,            
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout)
        self.inner_lr = 0.001
        self.outer_lr = 0.0001
        self.episode_num = 1000
        self.task_num_per_episode = 2
        self.valid_task_num_per_episode = 2
        self.inner_steps = 2
        self.device = 'cuda'
        self.models = self.build_model()
        self.models[0] = self.models[0].to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.maml = l2l.algorithms.MAML(self.models[0], lr=self.inner_lr, first_order=False, allow_unused=True)
        self.opt = optim.Adam(self.maml.parameters(), self.outer_lr)
        self.split_method = split_method
        self.split_frac = split_frac
        self.current_inner_lr = 0.001
        self.lr_increment = 0.00001
        self.logger.info(self.models[0])
        self.similarity_sequence = 0
        # self.fai_alpha = 0.002
        # self.logger.info(self.optimizers[0])

    def build_model(self):
        models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        return models

    def get_inner_lr(self):
        self.current_inner_lr += self.lr_increment
        return self.current_inner_lr

    def _get_data_loader(self, dataset, shuffle=False):
        return torch_geometric.loader.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                )

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (data, df) = self.dataset.get_split(return_df=True)
            self._task_data_df_split = (data, df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
            test_df=None, esb_yp=None, recalib_df=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if esb_yp is not None:
            if self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def cal_accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def fast_adapt(self, adaptation_data1, adaptation_data2, adaptation_labels, evaluation_data1, evaluation_data2,
                   evaluation_labels, learner, loss,
                   adaptation_steps, device):
        # torch.autograd.set_detect_anomaly(True)
        adaptation_data1, adaptation_data2, adaptation_labels, evaluation_data1, evaluation_data2, evaluation_labels = adaptation_data1.to(
            device), adaptation_data2.to(device), adaptation_labels.to(device), evaluation_data1.to(
            device), evaluation_data2.to(device), evaluation_labels.to(device)
        for step in range(adaptation_steps):
            s = learner(adaptation_data1, adaptation_data2)
            s = s.squeeze(1)
            train_error = loss(s, adaptation_labels)
            learner.adapt(train_error)
        predictions = learner(evaluation_data1, evaluation_data2)
        self.predictions = predictions.squeeze()
        valid_error = loss(self.predictions, evaluation_labels)
        return valid_error

    def process_data(self, data):
        all_feature_sequence = []
        for item in data:
            node_s = item.node_s
            node_v = item.node_v
            edge_s = item.edge_s
            edge_v = item.edge_v
            node_v_flattened = node_v.view(node_v.size(0), -1)
            node_features_combined = torch.cat((node_s, node_v_flattened), dim=1)
            edge_v_flattened = edge_v.view(edge_v.size(0), -1)
            edge_features_combined = torch.cat((edge_s, edge_v_flattened), dim=1)
            node_features_mean = node_features_combined.mean(dim=0, keepdim=True)
            edge_features_mean = edge_features_combined.mean(dim=0, keepdim=True)
            protein_features = torch.cat((node_features_mean, edge_features_mean), dim=1)
            all_feature_sequence.append(protein_features)
        return all_feature_sequence

    def cal_all_tasks_classes(self, data):
        all_tasks_classes = []
        for index, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(data):
            S_y_floats = [float(y) for y in S_y]
            items = [S_y_floats, S_xd, S_xp, Q_xd, Q_xp, Q_y]
            paired_items = list(zip(*items))
            sorted_pairs = sorted(paired_items, key=lambda x: x[0])
            mid_index = len(sorted_pairs) // 2
            class1 = sorted_pairs[:mid_index]
            class2 = sorted_pairs[mid_index:]
            S_y_class1 = [item[0] for item in class1]
            S_y_class2 = [item[0] for item in class2]
            S_xd_class1 = [item[1] for item in class1]
            S_xd_class2 = [item[1] for item in class2]
            S_xp_class1 = [item[2] for item in class1]
            S_xp_class2 = [item[2] for item in class2]
            task_classes = [
                [S_xd_class1, S_xp_class1, S_y_class1],
                [S_xd_class2, S_xp_class2, S_y_class2]
            ]
            all_tasks_classes.append(task_classes)
        return all_tasks_classes

    def cal_similarity(self, data):
        average_similarity_sequence = []
        all_mean_vector = []
        for task_classes in data:
            task_similarity_sequence = []
            for class_group in task_classes:
                #In the drug construction task, it should be changed to class_group[1]
                feature_sequence = self.process_data(class_group[1])
                stacked_tensors = torch.stack(feature_sequence, dim=0)
                for i in range(len(feature_sequence)):
                    for j in range(i + 1, len(feature_sequence)):
                        tensor_i = feature_sequence[i][0].unsqueeze(0)
                        tensor_j = feature_sequence[j][0].unsqueeze(0)
                        similarity = F.cosine_similarity(tensor_i, tensor_j, dim=1).squeeze()
                        task_similarity_sequence.append(similarity.item())
                if task_similarity_sequence:
                    average_similarity = torch.tensor(task_similarity_sequence).mean()
                    average_similarity_sequence.append(average_similarity.item())
            mean_vector = stacked_tensors.mean(dim=0)
            all_mean_vector.append(mean_vector)
        return average_similarity_sequence, all_mean_vector

    def cal_pai(self, data):
        differences = []
        for i in range(0, len(data) - 1, 2):
            difference = abs(data[i] - data[i + 1])
            differences.append(difference)
        return differences

    def cal_meta_train_error(self, task_batch, tasks, t_com, t_rel):
        meta_train_error = 0.0
        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):
            S_y = torch.tensor(S_y)
            variance = torch.var(S_y)
            variance = variance.unsqueeze(0)
            variance = variance.to(self.device)
            t_concatenate = torch.cat((t_rel[i], t_com[i], variance))
            s = self.maml.module.fai_t(t_concatenate)
            alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            S_xd = next(iter(DataLoader(S_xd, batch_size=len(S_xd))))
            S_xp = next(iter(DataLoader(S_xp, batch_size=len(S_xp))))
            S_y = torch.tensor(S_y)
            Q_xd = next(iter(DataLoader(Q_xd, batch_size=len(Q_xd))))
            Q_xp = next(iter(DataLoader(Q_xp, batch_size=len(Q_xp))))
            Q_y = torch.tensor(Q_y)
            self.maml.set_lr(alpha)
            learner = self.maml.clone()
            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y,
                                               learner,
                                               self.criterion,
                                               self.inner_steps,
                                               self.device)
            meta_train_error = evaluation_error + meta_train_error
        return meta_train_error

    def cal_meta_valid_error(self, task_batch, tasks, t_com, t_rel):
        meta_valid_error = 0.0
        all_Q_y = []
        all_predictions = []
        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):
            S_y = torch.tensor(S_y)
            variance = torch.var(S_y)
            variance = variance.unsqueeze(0)
            variance = variance.to(self.device)
            t_concatenate = torch.cat((t_rel[i], t_com[i], variance))
            s = self.maml.module.fai_t(t_concatenate)
            alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            S_xd = next(iter(DataLoader(S_xd, batch_size=len(S_xd))))
            S_xp = next(iter(DataLoader(S_xp, batch_size=len(S_xp))))
            S_y = torch.tensor(S_y)
            Q_xd = next(iter(DataLoader(Q_xd, batch_size=len(Q_xd))))
            Q_xp = next(iter(DataLoader(Q_xp, batch_size=len(Q_xp))))
            Q_y = torch.tensor(Q_y)
            self.maml.set_lr(alpha)
            learner = self.maml.clone()
            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y,
                                               learner,
                                               self.criterion,
                                               self.inner_steps,
                                               self.device)
            meta_valid_error = evaluation_error + meta_valid_error
            all_Q_y.append(Q_y.cpu().numpy())
            all_predictions.append(self.predictions.detach().cpu().numpy())
        all_Q_y = np.concatenate(all_Q_y, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        ci = concordance_index(all_Q_y, all_predictions)
        spearman_correlation, _ = spearmanr(all_Q_y, all_predictions)
        person_correlation, _ = pearsonr(all_Q_y, all_predictions)
        return ci,  person_correlation

    def cal_meta_test_error(self, task_batch, tasks, t_com, t_rel):
        all_Q_y = []
        all_predictions = []
        self.opt.zero_grad()
        meta_adapting_error = 0.0
        best_model_path = 'best_model.pth'
        self.maml.load_state_dict(torch.load(best_model_path, map_location=self.device))
        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):
            S_y = torch.tensor(S_y)
            variance = torch.var(S_y)
            variance = variance.unsqueeze(0)
            variance = variance.to(self.device)
            t_concatenate = torch.cat((t_rel[i], t_com[i], variance))
            s = self.maml.module.fai_t(t_concatenate)
            alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            S_xd = next(iter(DataLoader(S_xd, batch_size=len(S_xd))))
            S_xp = next(iter(DataLoader(S_xp, batch_size=len(S_xp))))
            S_y = torch.tensor(S_y)
            Q_xd = next(iter(DataLoader(Q_xd, batch_size=len(Q_xd))))
            Q_xp = next(iter(DataLoader(Q_xp, batch_size=len(Q_xp))))
            Q_y = torch.tensor(Q_y)
            self.maml.set_lr(alpha)
            learner = self.maml.clone()
            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y,
                                               learner,
                                               self.criterion,
                                               self.inner_steps,
                                               self.device)
            meta_adapting_error = evaluation_error + meta_adapting_error
            all_Q_y.append(Q_y.cpu().numpy())
            all_predictions.append(self.predictions.detach().cpu().numpy())
        all_Q_y = np.concatenate(all_Q_y, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_Q_y_tensor = torch.from_numpy(all_Q_y)
        all_predictions_tensor = torch.from_numpy(all_predictions)
        spearman_correlation, _ = spearmanr(all_Q_y, all_predictions)
        person_correlation, _ = pearsonr(all_Q_y, all_predictions)
        ci = concordance_index(all_Q_y, all_predictions)
        meta_test_error = self.criterion(all_predictions_tensor, all_Q_y_tensor)
        return ci,  person_correlation

    def cal_t_com(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(self.task_num_per_episode, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com


    def cal_t_com_valid(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(2, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com

    def cal_t_com_test(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(2, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com

    def cal_t_rel(self, t_com):
        gamma = 1
        globalpool_A = self.maml.module.globalpool_A
        c_ij = torch.randn((t_com.shape[0], globalpool_A.shape[0]), dtype=torch.float32).to(self.device)
        for i in range(t_com.shape[0]):
            for j in range(globalpool_A.shape[0]):
                distances_squared = (t_com[i].unsqueeze(0) - globalpool_A).pow(2).sum(dim=1)
                numerator = (1 + distances_squared / gamma) ** (-(gamma + 1) / 2)
                c_ij[i, :] = numerator / numerator.sum()
        t_rel = torch.matmul(c_ij, globalpool_A)
        return t_rel

    def maml_train(self):
        model_path = 'best_model.pth'
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"{model_path} has been deleted.")
        else:
            print(f"{model_path} does not exist.")
        best_spearman = -1
        best_ci = 0
        for iteration in range(self.episode_num):
            self.opt.zero_grad()
            train_task_batch, train_batch, tasks = self.dataset.get_batch_train_tasks(num_tasks=self.task_num_per_episode)
            t_com = self.cal_t_com(train_batch)
            t_rel = self.cal_t_rel(t_com)
            t_com_rel = torch.cat((t_com, t_rel), dim=1)
            # alpha = self.maml.module.fai_h(t_com_rel) * self.fai_alpha
            # alpha = self.maml.module.fai_t(t_com) * 0.002
            # s = self.maml.module.fai_t(t_rel)
            # alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
            # alpha = torch.mul(s, alpha_tensor)
            meta_train_error = self.cal_meta_train_error(train_task_batch, train_batch, t_com, t_rel)
            # meta_train_error = self.cal_meta_train_error(train_task_batch)
            print('Iteration: ', iteration, ' Meta Train Error: ', meta_train_error.item() / self.task_num_per_episode)
            meta_train_error = meta_train_error / self.task_num_per_episode
            meta_train_error.backward()
            self.opt.step()
            valid_task_num = 10
            if iteration >= 990:
                all_meta_test_error = 0
                all_spearman_correlation = 0
                all_ci = 0
                all_person_correlation = 0
                for _ in range(valid_task_num):
                    train_task_batch, valid_batch, tasks = self.dataset.get_batch_valid_tasks(
                        num_tasks=self.valid_task_num_per_episode)
                    t_com = self.cal_t_com_valid(valid_batch)
                    t_rel = self.cal_t_rel(t_com)
                    # t_com_rel = torch.cat((t_com, t_rel), dim=1)
                    # alpha = self.maml.module.fai_h(t_com_rel) * self.fai_alpha
                    # alpha = self.maml.module.fai_t(t_com) * self.maml.module.fai_alpha
                    # s = self.maml.module.fai_t(t_rel)
                    # alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
                    # alpha = torch.mul(s, alpha_tensor)
                    ci, person_correlation = self.cal_meta_valid_error(
                        train_task_batch, valid_batch, t_com, t_rel
                        )
                    # all_meta_test_error += meta_valid_error.item()
                    # all_spearman_correlation += spearman_correlation
                    all_ci += ci
                    all_person_correlation += person_correlation
                error = all_meta_test_error / valid_task_num / self.valid_task_num_per_episode
                ci = all_ci / valid_task_num
                spearman = all_spearman_correlation / valid_task_num
                person = all_person_correlation / valid_task_num
                print('-------------Verification result---------------')
                # print('MSE：', error)
                # print('spearman：', spearman)
                print('ci：', ci)
                print('pearson：', person)
                learner = self.maml.clone()
                if ci > best_ci:
                    best_ci = ci
                    best_model_state_dict = learner.state_dict()
                    print(f'New best model with ci {best_ci:.4f} saved.')
                    torch.save(best_model_state_dict, 'best_model.pth')

    def maml_test(self):
        self.opt.zero_grad()
        best_model_path = 'best_model.pth'
        self.maml.load_state_dict(torch.load(best_model_path, map_location=self.device))
        all_meta_test_error = 0
        all_spearman_correlation = 0
        all_person_correlation = 0
        all_ci = 0
        testing_task_num = 20
        for _ in range(testing_task_num):
            task_test_batch, test_batch, tasks = self.dataset.get_batch_test_tasks()
            t_com = self.cal_t_com_test(test_batch)
            t_rel = self.cal_t_rel(t_com)
            # t_com_rel = torch.cat((t_com, t_rel), dim=1)
            # alpha = self.maml.module.fai_h(t_com_rel) * self.fai_alpha
            # alpha = self.maml.module.fai_t(t_com) * self.maml.module.fai_alpha
            # s = self.maml.module.fai_t(t_rel)
            # alpha_tensor = torch.stack([alpha_param for alpha_param in self.maml.module.fai_alpha])
            # alpha = torch.mul(s, alpha_tensor)
            ci, person_correlation = self.cal_meta_test_error(task_test_batch, test_batch, t_com, t_rel
                                                                                            )
            # all_meta_test_error += meta_test_error.item()
            # all_spearman_correlation += spearman_correlation
            all_ci += ci
            all_person_correlation += person_correlation
        print('-------------Testing result---------------')
        # print('MSE：', all_meta_test_error / testing_task_num)
        print('ci: ', all_ci / testing_task_num)
        # print('spearman：', all_spearman_correlation / testing_task_num)
        print('pearson：', all_person_correlation / testing_task_num)

    def train(self, n_epochs=None, patience=None,
              eval_freq=1, test_freq=None,
              monitoring_score='pearson',
              train_data=None, valid_data=None,
              rebuild_model=False,
              test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx=i + 1,
                model=self.models[i],
                optimizer=self.optimizers[i],
                device=self.devices[i],
                train_loader=tl,
                valid_loader=vl,
                stopper=stp,
                best_model_state_dict=None,
            )
            rets_list.append(rets)
        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']

    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)
        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }
        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'spearman', 'pearson']
        )
        results['metrics'] = eval_metrics
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results

