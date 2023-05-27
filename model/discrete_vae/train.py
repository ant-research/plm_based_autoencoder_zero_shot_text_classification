"""
用来训练单个vq vae+GAN的架构
"""

import torch
import os
from tqdm import tqdm
import time
from codes.utils import save_data
from model.discrete_vae.models.model import Model
from model.new_model.modules.discriminator import Discriminator
from torch.nn.utils.clip_grad import clip_grad_norm_
from model.train_base import Trainbasic
from model.new_model.utils.loss import ContrastiveLoss, ContrastiveLoss_Neg
from model.new_model.train_split import OneModelTrainer_VQVAESplit
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed as data_dist

world_size = 8
cpu_num = max(1, int(cpu_count()/world_size) - 1) # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class OneModelTrainer_DisVAE(Trainbasic):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset,
                 emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        # 训练组件设置
        self.output_generate = model_config.output_generate
        self.unlabel = model_config.is_unlabel_train  # 是否使用unlabel dataset训练
        # 学习参数
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.div_batch_size = model_config.batch_size
        self.lr = model_config.lr
        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        # load dataset
        self.load_dataset(dataloader_list=dataloader_list)

        # build a new model
        self.build_model()
        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self):
        '''
        initialize model
        '''
        self.model = Model(config=self.model_config,
                           emb_layer=self.emb_layer)
        self.model.to(device)
        self.logger.info(self.model)

    def load_dataset(self, dataloader_list, **kwargs):
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset = dataloader_list
        self.train_loader = DataLoader(dataset=self.train_loader,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_loader,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.eval_loader = DataLoader(dataset=self.eval_loader,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)

    def train(self):
        """
        训练传入的模型
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))
        # start epoch
        for epoch in range(self.epochs):
            self.epoch = epoch

            # set data loader and get train setting
            if self.unlabel and epoch >= self.model_config.unlabel_epoch:
                unlabel = True
            else:
                unlabel = False

            # generator train
            gen_loss_dict, final_targets, final_probabs = self.generate_train(generate=False, unlabel=unlabel)
            vq_loss = gen_loss_dict['vq_loss']
            print('epoch', self.epoch, 'recon loss', gen_loss_dict['recon_loss'], 'vq loss', vq_loss)

    def batch_gen_train(self, batch, final_targets, final_probabs, is_train=True):
        '''
        common training
        '''
        # return loss
        total_loss = torch.tensor(0)
        vq_loss = 0
        recon_loss = 0

        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx'].to(device)
        # padding_list.append(result_dict['origin']['pad'].unsqueeze(0).to(device))
        # pad_mask = torch.cat(padding_list, dim=0).to(device)

        # get result
        result_dict = self.model(batch_x)
        # get da result
        # da_result_dict = self.model(batch_x_da)

        if is_train:
            # 检测生成的句子
            if self.output_generate and self.first_batch:
                sent_list, _, _ = self.generate_sentence(result_dict['pred_idx'], need_build=False)
                for i in range(len(sent_list)):
                    print('epoch:', self.epoch, 'recon句子', sent_list[i], 'origin sentence', text_list[i],
                          'p idx', result_dict['indices'][i])

            # start deal with loss
            # print('output loss', result_dict['loss'], result_dict['loss_commit'], result_dict['loss_reconstruct'])
            total_loss = result_dict['loss']
            total_loss = torch.sum(total_loss)/self.div_batch_size

            # get loss value
            recon_loss = torch.sum(result_dict['loss_reconstruct']).item()/self.div_batch_size
            vq_loss = total_loss.item() - recon_loss

        loss_dict = {
            'vq_loss': vq_loss,
            'recon_loss': recon_loss,
        }
        return loss_dict, final_targets, final_probabs, total_loss

    def unlabel_gen_train(self, batch):
        # 处理原文本以及da数据
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx'].to(device)

        # get result
        result_dict = self.model(batch_x)
        # da_result_dict = self.model(batch_x_da)

        # start deal with loss
        total_loss = result_dict['loss'] # + da_result_dict['loss']
        total_loss = torch.sum(total_loss)/self.div_batch_size

        return total_loss

    def generate_train(self, generate, unlabel):
        # init loss
        loss_dict = {
            'vq_loss': 0.0,
            'recon_loss': 0.0,
        }
        step = 0

        # start train, freeze discriminator and train generator

        self.logger.info('Start epoch %d' % self.epoch)
        final_targets = []
        final_probabs = []

        # first batch to generate
        if self.epoch % 50 == 0:
            self.first_batch = True
        length = min(len(self.train_loader), len(self.unlabel_loader))
        if generate is False and unlabel is False:
            data_loader = zip(self.train_loader)
        elif generate is True and unlabel is False:
            data_loader = zip(self.train_loader)
        else:
            assert unlabel is True
            data_loader = zip(self.train_loader, self.unlabel_loader)
        # start iterator
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            self.model.train()
            # start common batch
            train_batch = batch[0]
            b_loss_dict, final_targets, final_probabs, total_loss = self.batch_gen_train(batch=train_batch,
                                                                                         final_targets=final_targets,
                                                                                         final_probabs=final_probabs)
            # add to loss dict to return
            for key in loss_dict.keys():
                loss_dict[key] += b_loss_dict[key]

            if unlabel:
                unlabel_batch = batch[2]
                total_loss += self.unlabel_gen_train(unlabel_batch)

            # backward
            print('total loss is', total_loss)
            total_loss.backward()
            
            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(), self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False

        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/(step+1)
        
        return loss_dict, final_targets, final_probabs
        

class OneModelTrainerParallel_DisVAE(OneModelTrainer_DisVAE):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        dist.init_process_group('nccl')
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)
        world_size = dist.get_world_size()
        self.div_batch_size = model_config.batch_size/world_size
        
        
    def build_model(self):
        '''
        initialize model
        '''
        super().build_model()
        rank = dist.get_rank()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)

    def load_dataset(self, dataloader_list, **kwargs):
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset = dataloader_list
        train_sampler = data_dist.DistributedSampler(self.train_loader)
        test_sampler = data_dist.DistributedSampler(self.test_loader)
        dev_sampler = data_dist.DistributedSampler(self.eval_loader)
        unlabel_sampler = data_dist.DistributedSampler(self.unlabel_dataset)
        self.train_loader = DataLoader(dataset=self.train_loader,
                                       batch_size=self.batch_size,
                                       sampler=train_sampler)
        self.test_loader = DataLoader(dataset=self.test_loader,
                                      batch_size=self.batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_loader,
                                      batch_size=self.batch_size,
                                      sampler=dev_sampler)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.batch_size,
                                         sampler=unlabel_sampler)
        
    def train(self):
        """
        训练传入的模型
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Start running basic DDP example on rank {rank}.")
        # self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))
        # start epoch
        for epoch in range(self.epochs):
            self.train_loader.sampler.set_epoch(epoch)  # type: ignore
            self.epoch = epoch
            # set data loader and get train setting
            if self.unlabel and epoch >= self.model_config.unlabel_epoch:
                unlabel = True
            else:
                unlabel = False

            # generator train
            gen_loss_dict, _, _ = self.generate_train(generate=False, unlabel=unlabel)
            vq_loss = gen_loss_dict['vq_loss']
            print('epoch', self.epoch, 'recon loss', gen_loss_dict['recon_loss'], 'vq loss', vq_loss)