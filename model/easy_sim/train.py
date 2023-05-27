import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.train_base import Trainbasic
# from codes.utils import save_data, MemTracker, download_file
from multiprocessing import cpu_count

cpu_num = cpu_count() - 1  # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)


class OneModelTrainer_SIM(Trainbasic):
    '''
    训练模型
    '''
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None,
                 local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.generate_size = model_config.generate_size
        self.class_num = model_config.class_num
        # load dataset
        self.load_dataset(dataloader_list, generated_dataset)

        # deal with label matrix
        if label_mat.shape[1] == 1:
            label_mat = label_mat.squeeze(1)
        self.label_mat = label_mat

    def load_dataset(self, dataloader_list, generated_dataset):
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset, self.label_dataset = dataloader_list
        self.train_loader = DataLoader(dataset=self.train_loader,
                                       batch_size=self.batch_size,
                                       shuffle=False)
        self.test_loader = DataLoader(dataset=self.test_loader,
                                      batch_size=self.batch_size,
                                      shuffle=False)
        self.eval_loader = DataLoader(dataset=self.eval_loader,
                                      batch_size=self.batch_size,
                                      shuffle=False)

    def train(self):
        """
        训练传入的模型
        """
        pass

    def evaluate(self, loader, get_loss=False):
        final_targets = []
        final_probabs = []
        for batch in tqdm(self.test_loader):
            result_dict = batch
            batch_x_text = result_dict['origin']['text']
            batch_y = result_dict['y_data']
            batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
            # get result
            # batch_x = torch.mean(batch_x, dim=1)  # [batch_size, emb]
            batch_x = torch.max(batch_x, dim=1)[0]
            batch_x = batch_x/torch.norm(batch_x, p=2, dim=1, keepdim=True)
            label_mat = self.label_mat/torch.norm(self.label_mat, p=2, dim=1, keepdim=True)
            label_cos = torch.matmul(batch_x, label_mat.t().to(batch_x.device))
            print(batch_x.shape, label_cos.shape)

            batch_y = F.one_hot(batch_y, num_classes=self.class_num)
            final_targets.extend(batch_y.tolist())
            final_probabs.extend(F.softmax(label_cos, dim=1).detach().cpu().tolist())
        return final_probabs, final_targets, 0

    def test(self, name='test'):
        probabs, targets, _ = self.evaluate(loader=self.test_loader, get_loss=False)
        self.get_metric_result(probabs, targets, name=name)
        return probabs, targets


class OneModelTrainerParallel_SIM(OneModelTrainer_SIM):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None,
                 local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset,
                         emb_layer=emb_layer, local=local, **kwargs)
