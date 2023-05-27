import torch
from torch.utils.data import DataLoader
from model.new_model.utils.metric import compute_scores, compute_scores_single
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(14)  # 固定随机种子（CPU）
if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(14)  # 为当前GPU设置
    torch.cuda.manual_seed_all(14)  # 为所有GPU设置

print(device)


def build_dataset(dataset_list, index_list):
    '''
    build dataset use for k fold
    '''
    result = []
    for index in index_list:
        result += dataset_list[index]
    return result


class TaskTrainer(object):
    '''
    10 fold trainer for kesu data
    '''
    def __init__(self, model_config, logger, dataloader_list, model_trainer, label_mat, generated_dataset, emb_layer=None, local=False):
        super().__init__()
        self.logger = logger
        # configs
        self.model_config = model_config
        self.local = local
        self.batch_size = self.model_config.batch_size
        self.model_trainer = model_trainer

        # layers and tensors
        self.label_mat = label_mat
        self.emb_layer = emb_layer

        # dataset
        self.logger.info('start load dataset')
        self.load_data(dataloader_list)
        self.generated_dataset = generated_dataset

        # adjcent matrix
        self.parent_adj = self.dataset.parent_adj
        self.child_adj = self.dataset.child_adj

    def load_data(self, dataloader_list, k=10):
        '''
        K fold dataset split
        '''
        self.dataset = dataloader_list[0]
        self.unlabel_dataset = dataloader_list[1]
        self.labeled_dataset = dataloader_list[2]
        one_fold_len = int(len(self.dataset)/k)
        eval_len = len(self.dataset) - (k-1) * one_fold_len
        lengths = [one_fold_len for i in range(k-1)]
        lengths.append(eval_len)
        self.dataset_list = torch.utils.data.random_split(dataset=self.dataset,
                                                          lengths=lengths)
        self.kfold_list = []
        for i in range(k):
            test_index = i
            eval_index = i+1 if i !=  9 else 0
            train_index = [j for j in range(k)]
            train_index.remove(test_index)
            train_index.remove(eval_index)

            train_dataset = build_dataset(self.dataset_list, train_index)
            test_dataset = self.dataset_list[test_index]
            eval_dataset = self.dataset_list[eval_index]
            self.kfold_list.append([train_dataset, test_dataset, eval_dataset, self.unlabel_dataset, self.labeled_dataset])

    def train(self):
        self.logger.info('Run k fold')
        # level n 的结果
        total_probabs = []
        total_targets = []
        for i in range(len(self.kfold_list)):
            self.logger.info('Start fold %d' % i)
            dataloader_list = self.kfold_list[i]
            model_trainer = self.model_trainer(self.model_config,
                                               self.logger,
                                               dataloader_list=dataloader_list,
                                               label_mat=self.label_mat,
                                               generated_dataset=self.generated_dataset,
                                               emb_layer=self.emb_layer,
                                               parent_adj=self.parent_adj,
                                               child_adj=self.child_adj)
            model_trainer.train()
            probabs, targets = model_trainer.test()
            total_probabs.extend(probabs)
            total_targets.extend(targets)
        self.get_metric_result(total_probabs, total_targets, name='total test')

    def get_metric_result(self, probabs, targets, name='dev'):
        '''
        get metrics
        '''
        if self.model_config.task_type == 'Multi_Label':
            apk_n = compute_scores(probabs, targets, name=name)
        else:
            apk_n = compute_scores_single(probabs, targets, name=name)
        return apk_n


class TaskTrainerBERT(object):
    '''
    10 fold trainer for kesu data
    '''
    def __init__(self, model_config, logger, dataloader_list, model_trainer, label_mat, generated_dataset,
                 emb_layer_t=None, emb_layer_p=None, local=False):
        super().__init__()
        self.logger = logger
        # configs
        self.model_config = model_config
        self.local = local
        self.batch_size = self.model_config.batch_size
        self.model_trainer = model_trainer

        # layers and tensors
        self.label_mat = label_mat
        self.emb_layer_t = emb_layer_t
        self.emb_layer_p = emb_layer_p

        # dataset
        self.logger.info('start load dataset')
        self.load_data(dataloader_list)
        self.generated_dataset = generated_dataset

        # adjcent matrix
        self.parent_adj = self.dataset.parent_adj
        self.child_adj = self.dataset.child_adj


    def load_data(self, dataloader_list, k=10):
        '''
        K fold dataset split
        '''
        self.dataset = dataloader_list[0]
        self.unlabel_dataset = dataloader_list[1]
        self.labeled_dataset = dataloader_list[2]
        one_fold_len = int(len(self.dataset)/k)
        eval_len = len(self.dataset) - (k-1) * one_fold_len
        lengths = [one_fold_len for i in range(k-1)]
        lengths.append(eval_len)
        self.dataset_list = torch.utils.data.random_split(dataset=self.dataset,
                                                          lengths=lengths)
        self.kfold_list = []
        for i in range(k):
            test_index = i
            eval_index = i+1 if i != 9 else 0
            train_index = [j for j in range(k)]
            train_index.remove(test_index)
            train_index.remove(eval_index)

            train_dataset = build_dataset(self.dataset_list, train_index)
            test_dataset = self.dataset_list[test_index]
            eval_dataset = self.dataset_list[eval_index]
            self.kfold_list.append([train_dataset, test_dataset, eval_dataset, self.unlabel_dataset, self.labeled_dataset])

    def train(self):
        self.logger.info('Run k fold')
        # level n 的结果
        total_probabs = []
        total_targets = []
        for i in range(len(self.kfold_list)):
            self.logger.info('Start fold %d' % i)
            dataloader_list = self.kfold_list[i]
            model_trainer = self.model_trainer(self.model_config,
                                               self.logger,
                                               dataloader_list=dataloader_list,
                                               label_mat=self.label_mat,
                                               generated_dataset=self.generated_dataset,
                                               emb_layer_t=self.emb_layer_t,
                                               emb_layer_p=self.emb_layer_p,
                                               parent_adj=self.parent_adj,
                                               child_adj=self.child_adj,
                                               emb_layer =self.emb_layer_t)
            model_trainer.train()
            probabs, targets = model_trainer.test()
            total_probabs.extend(probabs)
            total_targets.extend(targets)
        self.get_metric_result(total_probabs, total_targets, name='total test')

    def get_metric_result(self, probabs, targets, name='dev'):
        '''
        get metrics
        '''
        if self.model_config.task_type == 'Multi_Label':
            apk_n = compute_scores(probabs, targets, name=name)
        else:
            apk_n = compute_scores_single(probabs, targets, name=name)
        return apk_n
