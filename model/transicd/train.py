import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from model.transicd.code.models import TransICD
from torch.utils.data import DataLoader
from model.train_base import Trainbasic
from codes.utils import save_data, MemTracker
from multiprocessing import cpu_count

cpu_num = cpu_count() - 1 # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class OneModelTrainer_TransICD(Trainbasic):
    '''
    训练模型
    '''
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        # learning params
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.generate_size = model_config.generate_size
        self.lr = model_config.lr

        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        # help tracker
        self.gpu_tracker = MemTracker()
        # load dataset
        self.load_dataset(dataloader_list, generated_dataset)

        # deal with label matrix
        if label_mat.shape[1] == 1:
            label_mat = label_mat.squeeze(1)
        self.label_mat = label_mat
        
        # build a new model
        self.build_model()
        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0.01)

    def build_model(self):
        '''
        initialize model
        '''
        self.gpu_tracker.track()
        self.model = TransICD(config=self.model_config,
                              device=device,
                              label_matrix=self.label_mat.to(device),
                              gpu_tracker=self.gpu_tracker)
        self.model.to(device)
        print('after model')
        self.gpu_tracker.track()

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
        self.logger.info('Start model train with name: %s' % (self.model_path))
        num_gpu = torch.cuda.device_count()

        best_loss = 0
        self.best_epoch = 0  # early stop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            self.logger.info('Start epoch %d' % epoch)
            for step, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True):
                result_dict = batch
                batch_x_text = result_dict['origin']['text']
                batch_y = result_dict['y_data'].to(device)
                batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
                # get result
                outputs, ldam_outputs, _ = self.model(batch_x, batch_y)
                # calculate loss
                if ldam_outputs is not None:
                    loss = self.get_loss(ldam_outputs, batch_y)
                else:
                    loss = self.get_loss(outputs, batch_y)

                # backward
                train_loss += torch.mean(loss).item()
                loss.sum().backward()
                self.optimizer.step()
                for param in self.model.parameters():
                    param.grad = None
            train_loss = train_loss/(step+1)

            # get metric result
            eval_loss = self.eval(name='dev')

            if eval_loss > best_loss:
                best_loss = eval_loss
                self.best_epoch = epoch
                state_dict = self.model.module.state_dict() if num_gpu > 1 else self.model.state_dict()
                self.logger.info('Save best model')
                torch.save({'state_dict': state_dict}, self.save_local_path)
                if self.local is False:
                    save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
                
            if epoch - self.best_epoch >= 10:
                break

    def evaluate(self, loader, get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            if get_loss:
                for batch in tqdm(self.eval_loader):
                    result_dict = batch
                    batch_x_text = result_dict['origin']['text']
                    batch_y = result_dict['y_data']
                    batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
                    # get result
                    outputs, ldam_outputs, _ = self.model(batch_x, batch_y)
                    # calculate loss
                    if ldam_outputs is not None:
                        loss = self.get_loss(ldam_outputs, batch_y)
                    else:
                        loss = self.get_loss(outputs, batch_y)
                    # backward
                    eval_loss += torch.mean(loss).item()
                    final_targets, final_probabs = self.get_prob_and_target(batch_y, outputs, final_targets, final_probabs)
                eval_loss = eval_loss/len(self.eval_loader.dataset)
            else:
                # Set the model to evaluation mode
                for batch in loader:
                    result_dict = batch
                    batch_x_text = result_dict['origin']['text']
                    batch_y = result_dict['y_data']
                    batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
                    # get result
                    outputs, ldam_outputs, _ = self.model(batch_x, batch_y)
                    final_targets, final_probabs = self.get_prob_and_target(batch_y, outputs, final_targets, final_probabs)
        print('max is', max(final_probabs))
        print('min is', min(final_probabs))
        # print(max(final_probabs), min(final_probabs))
        return final_probabs, final_targets, eval_loss

    def get_loss(self, outputs, batch_y):
        if self.model_config.task_type == 'Multi_Label':
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
        else:
            loss = F.cross_entropy(outputs, batch_y)
        return loss


class OneModelTrainerParallel_TransICD(OneModelTrainer_TransICD):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def build_model(self):
        '''
        initialize model
        '''
        initial_usage = torch.cuda.memory_allocated()
        print('initial usage', initial_usage)
        self.model = TransICD(config=self.model_config,
                              device=device,
                              label_matrix=self.label_mat.to(device),
                              gpu_tracker=self.gpu_tracker)
        self.model = torch.nn.DataParallel(self.model.to(device))
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)
