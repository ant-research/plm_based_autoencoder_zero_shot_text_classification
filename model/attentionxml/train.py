import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from collections import deque
from model.attentionxml.deepxml.networks import AttentionRNN
from model.attentionxml.deepxml.optimizers import DenseSparseAdam
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


class OneModelTrainer_AttnXML(Trainbasic):
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
        self.optimizer = DenseSparseAdam(self.model.parameters())
        # other special param
        self.swa_warmup = self.model_config.swa_warmup
        self.swa_step_num = 10

        # loss fun
        self.ce_loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # gradient clip
        self.gradient_clip_value, self.gradient_norm_queue = 5.0, deque([np.inf], maxlen=5)
        self.state = {}

    def build_model(self):
        '''
        initialize model
        '''
        self.gpu_tracker.track()
        self.model = AttentionRNN(model_config=self.model_config)
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

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def train(self):
        """
        训练传入的模型
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))
        num_gpu = torch.cuda.device_count()

        best_apk1 = 0
        self.best_epoch = 0  # early stop
        global_step = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            self.logger.info('Start epoch %d' % epoch)
            if epoch == self.swa_warmup:
                self.swa_init()
            for step, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True):
                result_dict = batch
                batch_x_text = result_dict['origin']['text']
                batch_y = result_dict['y_data'].to(device)

                batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
                lengths = x_pad_mask.sum(dim=1).to(device)
                pad_mask = (x_pad_mask != 0).to(device)
                print(lengths, lengths.max(), pad_mask)
                pad_mask = pad_mask[:, :int(lengths.max().item())]

                # get result
                scores = self.model(batch_x, lengths, pad_mask)
                # calculate loss
                loss = self.get_loss(scores, batch_y)
                # backward
                train_loss += torch.mean(loss).item()
                loss.sum().backward()
                self.clip_gradient()
                self.optimizer.step()
                for param in self.model.parameters():
                    param.grad = None
                # swa
                global_step += 1
                if global_step % self.swa_step_num == 0:
                    self.swa_step()
                    self.swap_swa_params()
                    # swa evaluation
                    apk_1 = self.eval(name='dev')
                    print('output result is', apk_1, best_apk1)
                    if apk_1 > best_apk1:
                        best_apk1, self.best_epoch = apk_1, epoch
                        state_dict = self.model.module.state_dict() if num_gpu > 1 else self.model.state_dict()
                        self.logger.info('Save best model')
                        torch.save({'state_dict': state_dict}, self.save_local_path)
                        if self.local is False:
                            save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
                    # save swa params
                    self.swap_swa_params()

            if epoch - self.best_epoch >= 30:
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
                    lengths = x_pad_mask.sum(dim=1).to(device)
                    pad_mask = (x_pad_mask != 0).to(device)
                    pad_mask = pad_mask[:, :int(lengths.max().item())]
                    scores = self.model(batch_x, lengths, pad_mask)
                    # backward
                    final_targets, final_probabs = self.get_prob_and_target(batch_y, scores, final_targets, final_probabs)
            else:
                # Set the model to evaluation mode
                for batch in loader:
                    result_dict = batch
                    batch_x_text = result_dict['origin']['text']
                    batch_y = result_dict['y_data']
                    batch_x, x_pad_mask = self.emb_data(batch_x_text, get_idx=False)
                    # get result
                    lengths = x_pad_mask.sum(dim=1).to(device)
                    pad_mask = (x_pad_mask != 0).to(device)
                    pad_mask = pad_mask[:, :int(lengths.max().item())]
                    scores = self.model(batch_x, lengths, pad_mask)
                    final_targets, final_probabs = self.get_prob_and_target(batch_y, scores, final_targets, final_probabs)
        print('max is', max(final_probabs))
        print('min is', min(final_probabs))
        # print(max(final_probabs), min(final_probabs))
        return final_probabs, final_targets, eval_loss

    def get_loss(self, outputs, batch_y):
        if self.model_config.task_type == 'Multi_Label':
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
        else:
            loss = self.ce_loss_fn(outputs, batch_y)
        return loss

    def swa_init(self):
        if 'swa' not in self.state:
            self.logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.clone().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n], p.data

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']


class OneModelTrainerParallel_AttnXML(OneModelTrainer_AttnXML):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def build_model(self):
        '''
        initialize model
        '''
        initial_usage = torch.cuda.memory_allocated()
        print('initial usage', initial_usage)
        self.model = AttentionRNN(model_config=self.model_config)
        self.model = torch.nn.DataParallel(self.model.to(device))
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)
