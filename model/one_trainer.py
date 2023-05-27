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


class OneTaskTrainer(object):
    '''
    10 fold trainer for kesu data
    '''
    def __init__(self, model_config, logger, dataloader_list, model_trainer, label_mat, generated_dataset, emb_layer_t=None, emb_layer_p=None, local=False, **kwargs):
        super().__init__()
        self.logger = logger
        # configs
        self.model_config = model_config
        self.local = local
        self.batch_size = self.model_config.batch_size
        self.model_trainer = model_trainer

        # layers and tensors
        self.label_mat = label_mat
        
        # 如果是bert，则两个encoder不同，否则完全相同
        self.emb_layer = emb_layer_t
        self.emb_layer_p = emb_layer_p

        # dataset
        self.logger.info('start load dataset')
        self.load_data(dataloader_list)
        self.generated_dataset = generated_dataset

        # adjcent matrix
        self.parent_adj = self.train_dataset.parent_adj
        self.child_adj = self.train_dataset.child_adj

    def load_data(self, dataloader_list, k=10):
        '''
        K fold dataset split
        '''
        self.train_dataset = dataloader_list[0]
        self.test_dataset = dataloader_list[1]
        self.eval_dataset = dataloader_list[2]
        self.unlabel_dataset = dataloader_list[3]
        self.dataloader_list = dataloader_list


    def train(self):
        self.logger.info('Run one train')
        # level n 的结果
        total_probabs = []
        total_targets = []

        model_trainer = self.model_trainer(self.model_config,
                                            self.logger,
                                            dataloader_list=self.dataloader_list,
                                            label_mat=self.label_mat,
                                            generated_dataset=self.generated_dataset,
                                            emb_layer=self.emb_layer,
                                            emb_layer_p=self.emb_layer_p,
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

