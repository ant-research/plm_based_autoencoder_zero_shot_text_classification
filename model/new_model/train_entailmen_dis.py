#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_entailmen_dis.py
@Time    :   2023/03/22 01:26:41
@Desc    :   ZeroAE codes
'''

# here put the import lib

from typing import List
import torch
import os
import random
import numpy as np
import time
from tqdm import tqdm
import transformers

from model.new_model.augmentation import eda
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed as data_dist
from torch.utils.data import DataLoader

from codes.utils import save_data, MemTracker, download_file
from model.new_model.model_trainer import NewModelTrainerBasic
from model.new_model.utils.loss import ContrastiveLoss

from model.new_model.model.entail_model import EntailmentModel, GenerateModel
from model.new_model.modules.discriminator import Discriminator
from model.new_model.utils.memory import DisperseMemory, DatasetMemory

from multiprocessing import cpu_count

cpu_num = cpu_count() - 1 # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'
# os.environ['NCCL_IB_DISABLE'] = '1'
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)


class BuildLoad_Mixin():
    """
    Build model and load dataset
    """
    def __init__(self, **kwargs) -> None:
        pass

    def build_model(self):
        '''
        initialize model
        '''
        # the bert and classifier model
        self.model = EntailmentModel(label_matrix=self.label_mat,
                            config=self.model_config,
                            device=device,
                            gpu_tracker=self.gpu_tracker,
                            emb_layer=self.emb_layer,
                            emb_layer_p=self.emb_layer_p,
                            adj_parent=self.parent_adj,
                            adj_child=self.child_adj)
        self.model.to(device)
        # the vq and generator model
        self.generator = GenerateModel(label_matrix=self.label_mat,
                                       config=self.model_config,
                                       device=device,
                                       gpu_tracker=self.gpu_tracker,
                                       emb_layer=self.emb_layer,
                                       emb_layer_p=self.emb_layer_p,
                                       adj_parent=self.parent_adj,
                                       adj_child=self.child_adj)
        self.generator.to(device)
        # the discriminator model
        self.discriminator = Discriminator[self.model_config.discri_type](config=self.model_config)
        self.discriminator.to(device)
        self.discriminator.eval()
        print('after model')


    def load_dataset(self, dataloader_list, generated_dataset, **kwargs):
        self.train_dataset, self.test_dataset, self.eval_dataset, self.unlabel_dataset, self.label_dataset = dataloader_list
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        self.label_loader = DataLoader(dataset=self.label_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        self.generated_dataset = generated_dataset
        self.generated_loader = DataLoader(dataset=self.generated_dataset,
                                           batch_size=self.generated_dataset,
                                           shuffle=True)


class Parallel_Trainer_Mixin(BuildLoad_Mixin):
    """
    Use to parallel train the model

    Args:
        BuildLoad_Mixin (_type_): _description_
    """
    def __init__(self, **kwargs) -> None:
        pass

    def build_model(self):
        '''
        Model to pytorch DDP
        '''
        super().build_model()
        rank = dist.get_rank()
        
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank], find_unused_parameters=False)
        
        self.generator = torch.nn.parallel.DistributedDataParallel(
            self.generator, device_ids=[rank], find_unused_parameters=False)
        
        self.discriminator = torch.nn.parallel.DistributedDataParallel(
            self.discriminator, device_ids=[rank], find_unused_parameters=False)
        after_init_linear1 = torch.cuda.memory_allocated()
        

    def load_dataset(self, dataloader_list, generated_dataset, **kwargs):
        """
        Dataset to DDP model

        Args:
            dataloader_list (_type_): _description_
            generated_dataset (_type_): _description_
        """
        print('using parallel load_dataset mixin')
        world_size = dist.get_world_size() # get world size
        
        # div the batch size by world size
        self.div_batch_size = self.batch_size
        self.batch_size = self.batch_size * world_size
        self.div_generate_size = self.generate_size
        self.generate_size = self.generate_size * world_size
        
        self.train_dataset, self.test_dataset, self.eval_dataset, self.unlabel_dataset, self.label_dataset = dataloader_list
        
        # get sampler
        train_sampler = data_dist.DistributedSampler(self.train_dataset)
        test_sampler = data_dist.DistributedSampler(self.test_dataset)
        dev_sampler = data_dist.DistributedSampler(self.eval_dataset)
        label_sampler = data_dist.DistributedSampler(self.label_dataset)
        unlabel_sampler = data_dist.DistributedSampler(self.unlabel_dataset)
        generated_sampler = data_dist.DistributedSampler(generated_dataset)
        
        print(len(self.test_dataset), len(self.eval_dataset), len(self.train_dataset), len(generated_dataset))
        # build dataloader, the single loader is used in the IUAS flow to drop some dataset
        self.single_test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.div_batch_size,
                                      shuffle=True)
        self.single_eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.div_batch_size,
                                      shuffle=True)
        
        # the multiloader for parallel train
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.div_batch_size,
                                       sampler=train_sampler)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=dev_sampler)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.div_batch_size,
                                         sampler=unlabel_sampler)
        self.label_loader = DataLoader(dataset=self.label_dataset,
                                         batch_size=self.div_batch_size,
                                         sampler=label_sampler)
        self.generated_loader = DataLoader(dataset=generated_dataset,
                                           batch_size=self.div_generate_size,
                                           sampler=generated_sampler)


class OneModelTrainer_Entailment_Dis(NewModelTrainerBasic):
    """
    the Entailment + Disentangled model trainer

    Args:
        NewModelTrainerBasic (_type_): _description_
    """
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset,
                 emb_layer=None, emb_layer_p=None, local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list=dataloader_list,
                         generated_dataset=generated_dataset,
                         emb_layer=emb_layer, emb_layer_p=emb_layer_p, 
                         local=local, label_mat=label_mat, **kwargs)

        # build a new model
        print('start build model')
        self.build_model()
        self.use_data_augmentation = True
        self.smooth_unlabeled_train = True
        self.balance_y_weight = True
        # build optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr)
        self.generator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                                    lr=self.lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.lr,
                                                        betas=(0.9, 0.999),
                                                        eps=1e-8,
                                                        weight_decay=0.01)

        # schduler
        if self.model_config.use_scheduler:
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=5, 
                                                                          num_training_steps=20000)
            
            self.generator_scheduler = transformers.get_linear_schedule_with_warmup(self.generator_optimizer, num_warmup_steps=1000, 
                                                                                    num_training_steps=1000)
       # loss function init
        self.contras_loss_fun = ContrastiveLoss().to(device)

        # 记住之前的distil和disperse组合
        self.pair_memory = {}
        for i in range(self.label_mat.shape[0]):
            self.pair_memory[i] = []
            
        # self training data sample
        self.init_fake_label_memory()
        self.choosed_topk = 1
        
        # fake label precision
        self.fake_label_true = [0 for _ in range(self.class_num)]
        self.fake_label_all = [0 for _ in range(self.class_num)]
        
        # disperse memory, used for disentanglement loss
        self.disperse_memory = DisperseMemory(self.class_num, max_len=100, disperse_max_len=self.model_config.disper_num)
        
        # dataset memory, used for generated train
        self.y_text_list = generated_dataset.y_text
        self.dataset_memory = DatasetMemory(self.y_text_list, self.model_config, self.emb_data, self.batch_size)
        
        # split string
        self.split_str = self.model_config.split_str
        
    def sample_class_by_weight(self, y:int, **kwargs):
        """
        Sample class by weight to balance the data

        Args:
            y (_type_): positive sample index
        """
        sampled_weight_list = [i for i in range(self.class_num) if i != y]
        sampled_y = random.sample(sampled_weight_list, 1)[0]
        return sampled_y

    def init_fake_label_memory(self):
        # inital fake label memory for pseudo label(if pseudo label is True)
        eval_index_list = self.eval_dataset.backup_index_list
        test_index_list = self.test_dataset.backup_index_list
        self.fake_labeled_data_idx_memory_eval = [-1 for i in eval_index_list]
        self.fake_labeled_data_idx_memory_test = [-1 for i in test_index_list]

    def store_fake_label(self, fake_label_index: int, fake_label_id: int, true_y_id: int, data_type: str):
        """
        Store the fake label data for pseudo labeling

        Args:
            fake_label_index (int): index of fake labeled data
            fake_label_id (int): fake label id of this data
            true_y_id (int): ture label id of this data
            data_type (int): data type: in test dataset or eval dataset
        """ 
        # self.logger.info('storing a fake label is %d, true label is %d' % (fake_label_id, true_y_id))
        if data_type == 'test':
            self.fake_labeled_data_idx_memory_test[fake_label_index] = fake_label_id
        elif data_type == 'eval':
            self.fake_labeled_data_idx_memory_eval[fake_label_index] = fake_label_id
        # calculate precision
        self.fake_label_all[fake_label_id] += 1
        if fake_label_id == true_y_id:
            self.fake_label_true[fake_label_id] +=1

    def find_fake_label(self, fake_label_index: int, data_type: str) -> int:
        """_summary_

        Args:
            fake_label_index (int): fake label index in it dataset
            data_type (str): 'test' or 'eval'
        """        
        if data_type == 'eval':
            return self.fake_labeled_data_idx_memory_eval[fake_label_index]
        elif data_type == 'test':
            return self.fake_labeled_data_idx_memory_test[fake_label_index]
        else:
            raise KeyError

    def data_augmentation(self, text):
        """EDA data augmentation

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        da_result = eda(text, num_aug=1)[0]
        return da_result

    def rebuild_dataset(self):
        """
        Rebuild dataset

        Returns:
            _type_: _description_
        """
        self.test_dataset.rebuild_no_fake_label_dataset(self.fake_labeled_data_idx_memory_test)
        self.eval_dataset.rebuild_no_fake_label_dataset(self.fake_labeled_data_idx_memory_eval)
        test_sampler = data_dist.DistributedSampler(self.test_dataset)
        dev_sampler = data_dist.DistributedSampler(self.eval_dataset)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=dev_sampler)
        self.eval_loader.sampler.set_epoch(self.epoch)  # type: ignore
        self.test_loader.sampler.set_epoch(self.epoch)  # type: ignore
        return len(self.test_dataset), len(self.eval_dataset)
        
    def rebuild_fake_labeled_dataset(self):
        self.test_dataset.rebuild_fake_label_dataset(self.fake_labeled_data_idx_memory_test)
        self.eval_dataset.rebuild_fake_label_dataset(self.fake_labeled_data_idx_memory_eval)
        test_sampler = data_dist.DistributedSampler(self.test_dataset)
        dev_sampler = data_dist.DistributedSampler(self.eval_dataset)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=dev_sampler)
        self.eval_loader.sampler.set_epoch(self.epoch)  # type: ignore
        self.test_loader.sampler.set_epoch(self.epoch)  # type: ignore

    def recover_dataset(self):
        self.test_dataset.rebuild_all_dataset()
        self.eval_dataset.rebuild_all_dataset()
        test_sampler = data_dist.DistributedSampler(self.test_dataset)
        dev_sampler = data_dist.DistributedSampler(self.eval_dataset)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.div_batch_size,
                                      sampler=dev_sampler)
        self.eval_loader.sampler.set_epoch(self.epoch)  # type: ignore
        self.test_loader.sampler.set_epoch(self.epoch)  # type: ignore

    def labeled_train(self):
        """
        Train Bert classifier and Generator with labeled data
        """ 
        self.model.train()
        self.generator.train()
        self.discriminator.eval()
        # init loss
        loss_dict = {
            'classifier_loss': 0.0,
        }
        
        final_targets = []
        final_probabs = []

        # 组合训练数据
        length = len(self.train_loader)
        data_loader = self.train_loader
        self.first_batch = True

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            final_targets, final_probabs, total_loss, generator_loss = self.batch_labeled_train(batch=batch,
                                                                                                final_targets=final_targets,
                                                                                                final_probabs=final_probabs)
    
            loss_dict['classifier_loss'] += total_loss.item()
            
            if self.model_config.detect:
                with torch.autograd.detect_anomaly():
                    # backward
                    # print('total loss is', total_loss)
                    total_loss.backward()
                    generator_loss.backward()
            else:
                total_loss.backward()
                generator_loss.backward()
                
            if self.first_batch:
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print('classifier', name)

            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(),
                                self.model_config.grad_norm)
                clip_grad_norm_(self.generator.parameters(),
                                self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()
            self.first_batch = False

            if self.model_config.use_scheduler:
                self.scheduler.step()
                self.generator_scheduler.step()
                
            if step >= 100:
                break

        # output result
        if dist.get_rank() == 0:
            curr_lr = self.scheduler.get_last_lr()
            output_str = 'now learning rate : %.4f' % curr_lr[0]
            for key in loss_dict.keys():
                loss_dict[key] = loss_dict[key]/(step+1)
                output_str += key + ': ' + str(loss_dict[key]) + ' '
            print('Generator epoch', self.epoch, output_str)

        return loss_dict, final_targets, final_probabs

    def batch_labeled_train(self, batch, final_targets, final_probabs, is_train=True):
        '''
        Labeled training, one batch
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        y = result_dict['y_data']
        batch_x_da = result_dict['da']['idx']
        
        # store postivie samples for generator train
        positive_sampled_x = [] # generator sentence index
        positive_sentence_x = [] # generator sentence
        generate_positive_x = []
        generate_negative_x = []
        da_positive_x = []
        da_negative_x = []

        assert len(batch_x) == y.shape[0]
        assert batch_x[0].shape[0] == self.class_num

        for i in range(y.shape[0]):
            label_id = y[i].item()
            # if dist.get_rank() == 0 and self.first_batch:
            #     print(text_list[0][i], batch_x[i][0, :], text_list[1][i], batch_x[i][1, :], len(text_list), 'label_id is', label_id)
            neg_label_id = self.sample_class_by_weight(label_id)
            positive_sample = batch_x[i][label_id, :]
            negative_sample = batch_x[i][neg_label_id, :]
            positive_sampled_x.append(positive_sample.unsqueeze(0))
            positive_sentence_x.append(text_list[label_id][i].replace('[SEP]', self.model_config.split_str))
            generate_positive_x.append(positive_sample.unsqueeze(0))
            generate_negative_x.append(negative_sample.unsqueeze(0))
            # data augumentation
            positive_da_sample = batch_x_da[i][label_id, :]
            negative_da_sample = batch_x_da[i][neg_label_id, :]
            da_positive_x.append(positive_da_sample.unsqueeze(0))
            da_negative_x.append(negative_da_sample.unsqueeze(0))

        generate_positive_x = torch.cat(generate_positive_x, dim=0)
        generate_negative_x = torch.cat(generate_negative_x, dim=0)
        da_positive_x = torch.cat(da_positive_x, dim=0)
        da_negative_x = torch.cat(da_negative_x, dim=0)
        positive_sampled_x = torch.cat(positive_sampled_x, dim=0).to(device)
                
        total_loss = self.train_model_by_input_idx(generate_positive_x, generate_negative_x, da_positive_x, da_negative_x, contrastive_learning=False)
        
        generator_result_dict = self.generator(positive_sampled_x,
                                               sentence=positive_sentence_x,
                                               only_encoder=False)
        # get generator loss
        generator_loss = generator_result_dict['loss']

        return final_targets, final_probabs, total_loss, generator_loss

    def emb_augment_data_by_input_sentence(self, sent_list, negative_sent_list):
        """Generate index of sentence and negative sentence

        Args:
            sent_list (_type_): _description_
            negative_sent_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 获取positive sample
        sent_list = [sent.replace(self.model_config.split_str, '[SEP]') for sent in sent_list]
        
        generate_positive_x, _ = self.emb_data(sent_list)
        generate_negative_x, _ = self.emb_data(negative_sent_list)
        
        da_sent_list = []
        da_neg_list = []
        for sent in sent_list:
            da_sent_list.append(self.data_augmentation(sent))
        for sent in negative_sent_list:
            da_neg_list.append(self.data_augmentation(sent))
        da_positive_x, _ = self.emb_data(da_sent_list)
        da_negative_x, _ = self.emb_data(da_neg_list)
        
        generate_positive_x = torch.tensor(generate_positive_x)
        generate_negative_x = torch.tensor(generate_negative_x)

        da_positive_x = torch.tensor(da_positive_x)
        da_negative_x = torch.tensor(da_negative_x)
        
    
        return generate_positive_x, generate_negative_x, da_positive_x, da_negative_x

    def train_model_by_input_idx(self, positive_x, negative_x, da_positive_x, da_negative_x, contrastive_learning=True):
        """
        Train the model with classification loss and contrastive loss

        Args:
            positive_x (_type_): _description_
            negative_x (_type_): _description_
            da_positive_x (_type_): _description_
            da_negative_x (_type_): _description_

        Returns:
            _type_: _description_
        """

        generate_y_pos = torch.ones(positive_x.shape[0], 1)
        generate_y_neg = torch.zeros(negative_x.shape[0], 1)
        generate_total_x = torch.cat([positive_x, negative_x], dim=0)
        generate_y = torch.cat([generate_y_pos, generate_y_neg], dim=0).to(device)


        da_total_x = torch.cat([da_positive_x, da_negative_x], dim=0)
        da_y_pos = torch.ones(da_positive_x.shape[0], 1)
        da_y_neg = torch.zeros(da_negative_x.shape[0], 1)
        da_y = torch.cat([da_y_pos, da_y_neg], dim=0).to(device)
        
        generate_total_x = torch.cat([generate_total_x, da_total_x], dim=0)
        generate_y = torch.cat([generate_y, da_y], dim=0)
        # print('total_x', generate_total_x.shape, 'generate y', generate_y.shape)
                    
        # # get result
        encoder_result_dict = self.model(generate_total_x.detach(),
                                        y=generate_y.detach()
                                        )
        total_loss = encoder_result_dict['loss']
            
        # # calculate contrastive loss
        if (self.model_config.is_contrastive_t) and (contrastive_learning is True):
            z_t_total = encoder_result_dict['z_t'].squeeze(1)
            one_batch_length = da_total_x.shape[0]
            z_t = z_t_total[:one_batch_length, :]
            z_t_da = z_t_total[one_batch_length:, :]
            # print(z_t.shape, z_t_da.shape)
            contrastive_loss_t = self.contras_loss_fun(z_t, z_t_da) * self.model_config.contrastive_t_coef
            total_loss = total_loss + contrastive_loss_t
            # total_loss_2.backward()

        return total_loss

    def unlabeled_train(self):
        """
        Train generator with unlabeled data
        """
        self.model.eval()
        self.generator.train()
        self.discriminator.eval()
        # init loss
        loss_dict = {
            'reconstruction loss': 0.0,
        }

        # 组合训练数据
        length = len(self.eval_loader)
        data_loader = zip(self.eval_loader, self.test_loader)
        self.first_batch = True

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            eval_loss = self.batch_unlabeled_train(batch=batch[0], data_type='eval')
            test_loss = self.batch_unlabeled_train(batch=batch[1], data_type='test')
            if step == 100:
                break
        return loss_dict    

    def batch_unlabeled_train(self, batch, data_type='eval'):
        '''
        One batch for unlabeled train
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        batch_data_idx_list = result_dict['index']
        batch_data_idx_list = [i.item() for i in batch_data_idx_list]
        y = result_dict['y_data']
        
        assert batch_x[0].shape[0] == self.class_num

        total_batch_x = []
        for i in range(len(batch_x)):
            total_batch_x.append(batch_x[i])

        total_batch_x = torch.cat(total_batch_x, dim=0)
        result_dict = self.model(total_batch_x.to(device))
        
        prob = result_dict['prob'].squeeze(1).unsqueeze(0)
        prob = prob.reshape(y.shape[0], -1)
        prob = F.softmax(prob, dim=1)

        fake_classes_list = prob.topk(3, dim=1)[1]

        
        gap_prob = 0.5
        smallest_prob = 0.9
        choosed_text = []
        choosed_x = []

        while smallest_prob >= 0.03:
            for i in range(len(batch_x)):
                # 如果已经存进fake label，则跳过
                if self.find_fake_label(batch_data_idx_list[i], data_type=data_type) != -1:
                    continue
                fake_labeles = [int(i.item()) for i in list(fake_classes_list[i])]
                # 判断是否存入fake label
                if prob[i, fake_labeles[0]] > self.pseudo_label_threshold and (prob[i, fake_labeles[0]] - prob[i, fake_labeles[1]]) > gap_prob:
                    self.store_fake_label(fake_label_index=batch_data_idx_list[i],
                                          fake_label_id=fake_labeles[0], 
                                          true_y_id=y[i].item(),
                                          data_type=data_type)
                else:
                    if prob[i, fake_labeles[0]] < smallest_prob:
                        pass
                    else:
                        texts = [text_list[j][i] for j in range(len(text_list))]
                        for fake_label in fake_labeles[:1]:
                            # fake_label = y[i].item()
                            choosed_text.append(texts[fake_label].replace('[SEP]', self.model_config.split_str))
                            choosed_x.append(batch_x[i][fake_label, :].unsqueeze(0))
            
            if len(choosed_text) > 0:
                break
            else:
                smallest_prob -= 0.05
        
        if len(choosed_text) == 0:
            return -1
        print('start train generator')
        choosed_x = torch.cat(choosed_x, dim=0)
        generator_result_dict = self.generator(choosed_x.to(device),
                                               sentence = choosed_text,
                                               only_encoder=False)
        
        # store to disperse list
        generate_indice = generator_result_dict['quantizer_out_p']['encoding_indices']
        
        # get generator loss
        generator_loss = generator_result_dict['loss']
        
        if self.model_config.detect:
            with torch.autograd.detect_anomaly():
                # backward
                print('generate loss is', generator_loss)
                generator_loss.backward()
            if self.first_batch:
                for name, param in self.generator.named_parameters():
                    if param.grad is None:
                        print('generator', name)
        else:
            generator_loss.backward()

        if self.model_config.grad_norm:
            clip_grad_norm_(self.generator.parameters(),
                            self.model_config.grad_norm)

        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()
        self.first_batch = False
        if self.model_config.use_scheduler:
            self.generator_scheduler.step()

        return generator_loss 

    def generate_train(self):
        """
        Train generator
        """
        self.model.train()
        self.generator.eval()
        self.discriminator.eval()
        # init loss
        loss_dict = {
            'classifier_loss': 0.0,
        }
        
        final_targets = []
        final_probabs = []

        # 组合训练数据
        length = len(self.generated_loader)
        data_loader = self.generated_loader

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            self.batch_generated_train(batch=batch,
                                       final_targets=final_targets,
                                       final_probabs=final_probabs)
            if self.model_config.use_scheduler:
                self.scheduler.step()

        return loss_dict, final_targets, final_probabs

    def first_epoch_generate_trian(self):
        """

        Args:
            batch (_type_): _description_
        """        
        # start epoch
        length = len(self.generated_loader)
        data_loader = self.generated_loader
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            y_idx, y, y_text, label_name = batch
            y_idx = y_idx.to(device)
            y = y.to(device)

            for i in range(y.shape[0]):
                one_y_idx = y_idx[i].unsqueeze(0)
                label_start_text = y_text[i] + self.model_config.add_str

                sent_list = [label_start_text + '' + label_name[i] for _ in range(self.model_config.disper_num)]
                sent_list = [sent.replace(' ', '') for sent in sent_list]
                
                # if self.output_generate and self.first_batch:
                if one_y_idx.item() in self.seen_train:
                    seen_type = 'train'
                elif one_y_idx.item() in self.seen_eval:
                    seen_type = 'eval'
                else:
                    seen_type = 'unseen'

                # 从句子中重新生成batch
                negative_sent_list = []
                for sent in sent_list:
                    sent = sent.split(self.model_config.split_str)
                    # if dist.get_rank() == 0:
                    #     print('sent 0 is', sent[0], 'y_text is,', y_text)
                    tmp_y_text = [text for text in y_text if text != sent[0]]
                    sent[0] = random.sample(tmp_y_text, 1)[0]
                    sent = '[SEP]'.join(sent)
                    negative_sent_list.append(sent)
                # 获取positive sample
                sent_list = [sent.replace(self.model_config.split_str, '[SEP]') for sent in sent_list]
                
                generate_positive_x, generate_negative_x, da_positive_x, da_negative_x = self.emb_augment_data_by_input_sentence(sent_list, negative_sent_list)
                
                total_loss = self.train_model_by_input_idx(generate_positive_x, generate_negative_x, da_positive_x, da_negative_x, contrastive_learning=False)
                total_loss.backward()

                if self.model_config.grad_norm:
                    clip_grad_norm_(self.model.parameters(),
                                    self.model_config.grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.first_batch = False          
            if self.model_config.use_scheduler:
                self.scheduler.step()

    def batch_generated_train(self, batch, final_probabs, final_targets):
        """
        Generated data training, one batch

        Args:
            batch (_type_): _description_
        """        
        # start epoch
        y_idx, y, y_text, label_name = batch
        y_idx = y_idx.to(device)
        y = y.to(device)

        for i in range(y.shape[0]):
            one_y_idx = y_idx[i].unsqueeze(0)
            label_start_text = y_text[i] + self.model_config.add_str
            if isinstance(self.generator, torch.nn.parallel.DistributedDataParallel):
                result_dict, p_idx = self.generator.module.sample_all(label_start_text, device=device, one_by_one=False)  # type: ignore
            else:
                result_dict, p_idx = self.generator.sample_all(label_start_text, device=device, one_by_one=False)
            sent_list = result_dict['sentences']
            
            # if self.output_generate and self.first_batch:
            if one_y_idx.item() in self.seen_train:
                seen_type = 'train'
            elif one_y_idx.item() in self.seen_eval:
                seen_type = 'eval'
            else:
                seen_type = 'unseen'

            # output sentences
            # for j in range(len(sent_list)):
            #     print('epoch ', self.epoch, 'label', y_text[i], p_idx[j], '--- generate 句子', sent_list[j], '--- label seen type', seen_type)
            # 从句子中重新生成batch
            negative_sent_list = []
            for i in range(len(sent_list)):
                # update generative dataset
                self.dataset_memory.update_one_sentence(sent_list[i], one_y_idx.item())
                
                sent = sent_list[i].split(self.model_config.split_str)
                sent = '[SEP]'.join(sent)
                sent_list[i] = sent
                tmp_y_text = [text for text in y_text if text != sent[0]]
                negative_sent = [random.sample(tmp_y_text, 1)[0], sent[1]]
                negative_sent = '[SEP]'.join(sent)
                negative_sent_list.append(negative_sent)
                
                
            # 获取positive sample
            sent_list = [sent.replace(self.model_config.split_str, '[SEP]') for sent in sent_list]

            generate_positive_x, generate_negative_x, da_positive_x, da_negative_x = self.emb_augment_data_by_input_sentence(sent_list, negative_sent_list)
            
            total_loss = self.train_model_by_input_idx(generate_positive_x, generate_negative_x, da_positive_x, da_negative_x, contrastive_learning=False)
            total_loss.backward()

            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(),
                                self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False  
    
    def fake_labeled_data_train(self, data_loader, data_type: str):
        """
        Pseudo labeled data train
        """
        self.model.train()
        self.generator.eval()
        # 组合训练数据
        length = len(data_loader)
        self.first_batch = True

        # start epoch
        total_batch_x = []
        choosed_y = []
        for step, result_dict in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
                    # deal batch data
            batch_x = result_dict['origin']['idx']
            batch_data_idx_list = result_dict['index']
            batch_data_idx_list = [i.item() for i in batch_data_idx_list]
            y = result_dict['y_data']
            
            assert batch_x[0].shape[0] == self.class_num

            
            for i in range(len(batch_x)):
                fake_label = self.find_fake_label(batch_data_idx_list[i], data_type=data_type)
                if fake_label == -1:
                    continue
                else:
                    choosed_y.append(fake_label)
                    total_batch_x.append(batch_x[i])
            
            if len(choosed_y) < self.div_batch_size:
                continue
            else:
                # choose positive and negative sample to train classifier
                sampled_x = []
                sampled_y = []
                
                assert len(total_batch_x) == len(choosed_y)
                assert total_batch_x[0].shape[0] == self.class_num

                for i in range(len(choosed_y)):
                    label_id = choosed_y[i]
                    neg_label_id = self.sample_class_by_weight(label_id, split=False)
                    positive_sample = total_batch_x[i][label_id, :]
                    negative_sample = total_batch_x[i][neg_label_id, :]
                    sampled_x.append(positive_sample.unsqueeze(0))
                    sampled_x.append(negative_sample.unsqueeze(0))
                    sampled_y.append(1.0)
                    sampled_y.append(0.0)
                
                sampled_x = torch.cat(sampled_x, dim=0).to(device)
                sampled_y = torch.tensor(sampled_y).to(device).unsqueeze(1)
                
                # get classifier result
                result_dict = self.model(sampled_x,
                                        y=sampled_y,
                                        )

                # calculate probability
                total_loss = result_dict['loss']
                
                if self.model_config.detect:
                    with torch.autograd.detect_anomaly():
                        # backward
                        print('total loss is', total_loss)
                        total_loss.backward()
                    if self.first_batch:
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print('classifier', name)
                else:
                    total_loss.backward()

                if self.model_config.grad_norm:
                    clip_grad_norm_(self.model.parameters(),
                                    self.model_config.grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.first_batch = False
                total_batch_x = []
                choosed_y = []

    def generated_fake_data_train(self, data_loader):
        """
        Train generator
        """
        self.model.train()
        self.generator.eval()
        # 组合训练数据
        length = len(data_loader)
        self.first_batch = True

        # start epoch
        for step, result_dict in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            text_list = result_dict['origin']['text']
            batch_x = result_dict['origin']['idx']
            y = result_dict['y_data']
            
            # store postivie samples for generator train
            positive_sampled_x = [] # generator sentence index
            positive_sentence_x = [] # generator sentence
            generate_positive_x = []
            generate_negative_x = []

            assert len(batch_x) == y.shape[0]
            assert batch_x[0].shape[0] == self.class_num

            for i in range(y.shape[0]):
                label_id = y[i].item()
                # if dist.get_rank() == 0 and self.first_batch:
                #     print(text_list[0][i], batch_x[i][0, :], text_list[1][i], batch_x[i][1, :], len(text_list), 'label_id is', label_id)
                neg_label_id = self.sample_class_by_weight(label_id)
                positive_sample = batch_x[i][label_id, :]
                negative_sample = batch_x[i][neg_label_id, :]
                positive_sampled_x.append(positive_sample.unsqueeze(0))
                positive_sentence_x.append(text_list[label_id][i].replace('[SEP]', self.model_config.split_str))
                generate_positive_x.append(positive_sample.unsqueeze(0))
                generate_negative_x.append(negative_sample.unsqueeze(0))
                # data augumentation

            generate_positive_x = torch.cat(generate_positive_x, dim=0)
            generate_negative_x = torch.cat(generate_negative_x, dim=0)
            positive_sampled_x = torch.cat(positive_sampled_x, dim=0).to(device)

            generate_y_pos = torch.ones(generate_positive_x.shape[0], 1)
            generate_y_neg = torch.zeros(generate_negative_x.shape[0], 1)
            generate_total_x = torch.cat([generate_positive_x, generate_negative_x], dim=0)
            generate_y = torch.cat([generate_y_pos, generate_y_neg], dim=0).to(device)
                        
            # # get result
            encoder_result_dict = self.model(generate_total_x.detach(),
                                            y=generate_y.detach()
                                            )
            total_loss = encoder_result_dict['loss']
            

            if self.model_config.detect:
                with torch.autograd.detect_anomaly():
                    # backward
                    print('total loss is', total_loss)
                    total_loss.backward()
            else:
                total_loss.backward()
                
            if self.first_batch:
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print('classifier', name)

            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(),
                                self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False
                
            if step >= 100:
                break

    def sample_disperse_vectors(self, idx_list):
        if isinstance(self.generator, torch.nn.parallel.DistributedDataParallel):
            emb_vector = self.generator.module.sample_vectors(idx_list)  # type: ignore
        else:
            emb_vector= self.generator.sample_vectors(idx_list)
        return emb_vector

    def discriminator_train(self):
        """freeze model and train discrminator

        Args:
            generate (_type_): _description_
            unlabel (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # start train, freeze discriminator and train generator
        self.model.eval()
        self.generator.eval()
        self.discriminator.train()
        # return dict

        # start epoch
        length = len(self.generated_loader)
        data_loader = self.generated_loader
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            y_idx, y, y_text, label_name = batch
            y_idx = y_idx.to(device)
            y = y.to(device)

            for i in range(y.shape[0]):
                one_y_idx = y_idx[i].unsqueeze(0)
                neg_idx_result = self.disperse_memory.sample_negative(index=one_y_idx)
                
                first_emb_list = [i[0] for i in neg_idx_result]
                second_emb_list = [i[1] for i in neg_idx_result]
                
                first_emb_p = self.sample_disperse_vectors(first_emb_list).to(device)
                second_emb_p = self.sample_disperse_vectors(second_emb_list).to(device)
                y_idx_tensor = torch.tensor([one_y_idx for _ in first_emb_list])
                y_idx_vector = F.one_hot(y_idx_tensor, num_classes=self.class_num).to(device)

                neg_inputs = torch.cat([y_idx_vector, first_emb_p.squeeze(1), second_emb_p.squeeze(1)], dim=1)
                neg_y = torch.zeros(len(first_emb_list))
                
                pos_idx_result = self.disperse_memory.sample_positive(index=one_y_idx)
                first_emb_list = [i[0] for i in pos_idx_result]
                second_emb_list = [i[1] for i in pos_idx_result]

                first_emb_p = self.sample_disperse_vectors(first_emb_list).to(device).squeeze(1)
                second_emb_p = self.sample_disperse_vectors(second_emb_list).to(device).squeeze(1)
                y_idx_tensor = torch.tensor([one_y_idx for _ in first_emb_list])
                y_idx_vector = F.one_hot(y_idx_tensor, num_classes=self.class_num).to(device)
                pos_inputs = torch.cat([y_idx_vector, first_emb_p, second_emb_p], dim=1)
                pos_y = torch.ones(len(first_emb_list))
                inputs = torch.cat([pos_inputs, neg_inputs], dim=0)
                y = torch.cat([pos_y, neg_y], dim=0)
                
                result_dict = self.discriminator(
                    inputs = inputs.to(device),
                    label = y.to(device)
                )

                total_loss = result_dict['loss']
                total_loss.backward()

                self.discriminator_optimizer.step()
                self.discriminator_optimizer.zero_grad()
                self.first_batch = False   

    def seen_and_unseen_test(self, dataloader, data_type="test"):
        self.logger.info('Start model test with name: %s, epoch %d' % (self.model_path, self.best_epoch))
        if data_type == "test":
            try:
                self.model.load_state_dict(torch.load(self.save_local_path)['state_dict'])
            except RuntimeError:
                try:
                    self.model.module.load_state_dict(torch.load(self.save_local_path)['state_dict'])
                except:
                    self.logger.info('Use origin model because not found any save model state')
        else:
            pass
        
        seen_prob = []
        seen_targets = []
        unseen_prob = []
        unseen_targets = []
        self.logger.info('start test')
        
        if type(dataloader) == dict:
            probabs = []
            targets = []
            for one_data_type in dataloader.keys():
                one_loader = dataloader[one_data_type]
                one_probabs, one_targets, _ = self.evaluate(loader=one_loader, data_type=one_data_type, get_loss=False)
                probabs = probabs + one_probabs
                targets = targets + one_targets
                for i in range(len(targets)):
                    y = targets[i].index(1)
                    if y in self.seen_train:
                        seen_prob.append(probabs[i])
                        seen_targets.append(targets[i])
                    else:
                        unseen_prob.append(probabs[i])
                        unseen_targets.append(targets[i])
        else:
            probabs, targets, _ = self.evaluate(loader=dataloader, get_loss=False, data_type=data_type)
            for i in range(len(targets)):
                y = targets[i].index(1)
                if y in self.seen_train:
                    seen_prob.append(probabs[i])
                    seen_targets.append(targets[i])
                else:
                    unseen_prob.append(probabs[i])
                    unseen_targets.append(targets[i])

        self.get_metric_result(probabs, targets, name=f'{data_type} epoch {self.epoch} test total')
        self.get_metric_result(seen_prob, seen_targets, name=f'{data_type} epoch {self.epoch} test seen')
        self.get_metric_result(unseen_prob, unseen_targets, name=f'{data_type} epoch {self.epoch} test unseen')
        return probabs, targets

    def evaluate(self, loader, data_type:str ='eval', get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        self.generator.eval()
        eval_loss = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
                result_dict = batch
                batch_x = result_dict['origin']['idx']
                y = result_dict['y_data']
                
                assert len(batch_x) == y.shape[0]
                assert batch_x[0].shape[0] == self.class_num
                
                total_true_label = []
                total_batch_x = []
                for i in range(y.shape[0]):
                    true_label = [0.0 for _ in range(self.class_num)]
                    true_label[y[i].item()] = 1.0
                    total_true_label = total_true_label + true_label
                    total_batch_x.append(batch_x[i])
                # print('check x and true label shape', x.shape, true_label.shape)
                batch_x = torch.cat(total_batch_x, dim=0)
                true_label = torch.tensor(total_true_label).unsqueeze(1)
                result_dict = self.model(batch_x.to(device),
                                         y = true_label.to(device))
                prob = result_dict['prob'].squeeze(1).unsqueeze(0)
                prob = prob.reshape(y.shape[0], -1)
                # calculate loss
                eval_loss += torch.mean(result_dict['loss']).item()
                # prob = torch.cat(prob_list, dim=0)
                # print('output prob', prob, prob.shape, torch.tensor([y[i].item()]), torch.tensor([y[i].item()]).shape)
                final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)

        # print(max(final_probabs), min(final_probabs))
        return final_probabs, final_targets, eval_loss

    def train(self):
        """
        The main train flow
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))
        best_loss = 0
        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))

        # start epoch
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            # generator train
            if self.model_config.ablation_discriminator:
                self.discriminator_train()
            self.labeled_train()
            self.unlabeled_train()
            self.generate_train()
            
            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("evaluation precision reduce result %.4f" % ap_1.item())
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()  # type: ignore

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
        
        self.seen_and_unseen_test(dataloader=self.test_loader, data_type="test")

    def contrastive_train(self):
        """
        Get constractive loss
        """
        self.model.train()
        self.generator.eval()
        self.discriminator.eval()

        length = len(self.eval_loader)
        # data_loader = zip(self.eval_loader, self.test_loader)
        data_loader = zip(self.train_loader)
        self.first_batch = True

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            self.batch_contrastive_train(batch=batch[0])
            # self.batch_contrastive_train(batch=batch[1])
            if step == 100:
                break

    def batch_contrastive_train(self, batch):
        '''
        Contrastive train, one batch
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        y = result_dict['y_data']
        batch_x_da = result_dict['da']['idx']
        
        # store postivie samples for generator train
        positive_sampled_x = [] # generator sentence index
        positive_sentence_x = [] # generator sentence
        generate_positive_x = []
        generate_negative_x = []
        da_positive_x = []
        da_negative_x = []

        assert len(batch_x) == y.shape[0]
        assert batch_x[0].shape[0] == self.class_num

        for i in range(y.shape[0]):
            label_id = y[i].item()
            # if dist.get_rank() == 0 and self.first_batch:
            #     print(text_list[0][i], batch_x[i][0, :], text_list[1][i], batch_x[i][1, :], len(text_list), 'label_id is', label_id)
            neg_label_id = self.sample_class_by_weight(label_id)
            positive_sample = batch_x[i][label_id, :]
            negative_sample = batch_x[i][neg_label_id, :]
            positive_sampled_x.append(positive_sample.unsqueeze(0))
            positive_sentence_x.append(text_list[label_id][i].replace('[SEP]', self.model_config.split_str))
            generate_positive_x.append(positive_sample.unsqueeze(0))
            generate_negative_x.append(negative_sample.unsqueeze(0))
            # data augumentation
            positive_da_sample = batch_x_da[i][label_id, :]
            negative_da_sample = batch_x_da[i][neg_label_id, :]
            da_positive_x.append(positive_da_sample.unsqueeze(0))
            da_negative_x.append(negative_da_sample.unsqueeze(0))

        generate_positive_x = torch.cat(generate_positive_x, dim=0)
        generate_negative_x = torch.cat(generate_negative_x, dim=0)
        da_positive_x = torch.cat(da_positive_x, dim=0)
        da_negative_x = torch.cat(da_negative_x, dim=0)
        positive_sampled_x = torch.cat(positive_sampled_x, dim=0).to(device)

        generate_y_pos = torch.ones(generate_positive_x.shape[0], 1)
        generate_y_neg = torch.zeros(generate_negative_x.shape[0], 1)
        generate_total_x = torch.cat([generate_positive_x, generate_negative_x], dim=0)
        generate_y = torch.cat([generate_y_pos, generate_y_neg], dim=0).to(device)


        da_total_x = torch.cat([da_positive_x, da_negative_x], dim=0)
        da_y_pos = torch.ones(da_positive_x.shape[0], 1)
        da_y_neg = torch.zeros(da_negative_x.shape[0], 1)
        da_y = torch.cat([da_y_pos, da_y_neg], dim=0).to(device)
        
        generate_total_x = torch.cat([generate_total_x, da_total_x], dim=0)
        generate_y = torch.cat([generate_y, da_y], dim=0)
        # print('total_x', generate_total_x.shape, 'generate y', generate_y.shape)
                    
        # # get result
        encoder_result_dict = self.model(generate_total_x.detach(),
                                        y=generate_y.detach()
                                        )
        total_loss = encoder_result_dict['loss']
            
        # calculate contrastive loss
        z_t_total = encoder_result_dict['z_t'].squeeze(1)
        one_batch_length = da_total_x.shape[0]
        z_t = z_t_total[:one_batch_length, :]
        z_t_da = z_t_total[one_batch_length:, :]
        # print(z_t.shape, z_t_da.shape)
        contrastive_loss_t = self.contras_loss_fun(z_t, z_t_da) * self.model_config.contrastive_t_coef
        total_loss = total_loss * 0 + contrastive_loss_t
        # total_loss_2.backward()

        total_loss.backward()

        if self.model_config.grad_norm:
            clip_grad_norm_(self.model.parameters(),
                            self.model_config.grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.first_batch = False

    def label_unlabeled_data(self, data_type:str):
        """
        Label unlabeled data for pseudo label setting
        """
        self.model.eval()
        self.generator.eval()
        self.discriminator.eval()
        # 组合训练数据
        if data_type == 'eval':
            dataloader = self.eval_loader
        elif data_type == 'test':
            dataloader = self.test_loader
        else:
            raise KeyError

        length = len(dataloader)
        data_loader = dataloader
        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            result_dict = batch
            batch_x = result_dict['origin']['idx']
            batch_data_idx_list = result_dict['index']
            batch_data_idx_list = [i.item() for i in batch_data_idx_list]
            y = result_dict['y_data']
            
            # print(dist.get_rank(), 'start unlabel train')
            
            assert batch_x[0].shape[0] == self.class_num

            prob = self.calculate_batch_prob(batch_x)
            prob = F.softmax(prob, dim=1)
            # fake_class_list = torch.argmax(prob, dim=1)
            fake_classes_list = prob.topk(3, dim=1)[1]

            
            gap_prob = 0.5
            for i in range(len(batch_x)):
                # 如果已经存进fake label，则跳过
                if self.find_fake_label(batch_data_idx_list[i], data_type=data_type) != -1:
                    continue
                fake_labeles = [int(i.item()) for i in list(fake_classes_list[i])]
                # 判断是否存入fake label
                if prob[i, fake_labeles[0]] > self.pseudo_label_threshold or (prob[i, fake_labeles[0]] - prob[i, fake_labeles[1]]) > gap_prob:
                    # print('storing a fake label is', fake_labeles[0], 'true label is', y[i].item(), prob[i])
                    self.store_fake_label(fake_label_index=batch_data_idx_list[i],
                                        fake_label_id=fake_labeles[0], 
                                        true_y_id=y[i].item(),
                                        data_type=data_type)

    def calculate_batch_prob(self, batch_x: List):
        
        if self.class_num * len(batch_x) > 600:
            prob_result = []
            for i in range(len(batch_x)):
                result_dict = self.model(batch_x[i].to(device))
                prob_result.append(result_dict['prob'].squeeze(1).unsqueeze(0).detach().reshape(-1, self.class_num))
            prob = torch.cat(prob_result, dim=0)
            # print(prob.shape)
        else:
            total_batch_x = []
            for i in range(len(batch_x)):
                total_batch_x.append(batch_x[i])
            # print('check x and true label shape', x.shape, true_label.shape)
            total_batch_x = torch.cat(total_batch_x, dim=0)
            result_dict = self.model(total_batch_x.to(device))
            prob = result_dict['prob'].squeeze(1).unsqueeze(0).detach()
            prob = prob.reshape(-1, self.class_num)
        return prob


class OneModelTrainerParallel_Entailment_Dis(OneModelTrainer_Entailment_Dis, Parallel_Trainer_Mixin):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        dist.init_process_group('nccl')
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def train(self):
        """
        The main train flow
        """
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        # self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print(f'dataset length, train: {len(self.train_loader)}, unlabeled: {len(self.unlabel_loader)}')
        best_loss = 0
        self.best_epoch = 0
        # start epoch
        if self.model_config.detect is True:
            torch.autograd.set_detect_anomaly(True)            

        for epoch in range(self.max_epochs):
            self.train_loader.sampler.set_epoch(epoch)   # type: ignore
            self.eval_loader.sampler.set_epoch(epoch)  # type: ignore
            self.test_loader.sampler.set_epoch(epoch)  # type: ignore
            self.unlabel_loader.sampler.set_epoch(epoch)  # type: ignore
            self.generated_loader.sampler.set_epoch(epoch)  # type: ignore

            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            # generator train
            if self.model_config.ablation_discriminator:
                self.discriminator_train()
            if epoch == 0:
                for i in range(5):
                    self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                    self.generate_train()
            for i in range(3):
                self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                self.generate_train()
                self.unlabeled_train()

            self.fake_labeled_data_train(self.eval_loader, data_type='eval')
            self.fake_labeled_data_train(self.test_loader, data_type='test')

            self.logger.info('Start epoch %d, evaluation' % (self.epoch))
            output_str = "Epoch %d \n" % self.epoch
            if dist.get_rank() == 0:
                for i in range(len(self.fake_label_true)):
                    if self.fake_label_all[i] == 0:
                        precision = 0.00
                    else:
                        precision = self.fake_label_true[i]/self.fake_label_all[i]
                    output_str = output_str + "label: %d, precision %.4f \n" % (i, precision)
                self.logger.info(output_str)

            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("evaluation precision reduce result %.4f" % ap_1.item())
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
                    self.logger.info('Start test on best model')
                    self.seen_and_unseen_test(dataloader=[self.single_test_loader, self.single_eval_loader], data_type="unseen")
                    self.seen_and_unseen_test(dataloader=self.test_loader, data_type="test")
                    self.logger.info('End test on best model')
        
        self.seen_and_unseen_test(dataloader=[self.single_test_loader, self.single_eval_loader], data_type="test")


class OneModelTrainer_EntailmentImbalance_Dis(OneModelTrainer_Entailment_Dis, BuildLoad_Mixin):
    """
    Rebalance the data by choose the negative sample

    Args:
        OneModelTrainer_Entailment (_type_): _description_
    """
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset,
                 emb_layer=None, emb_layer_p=None, local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list=dataloader_list,
                         generated_dataset=generated_dataset,
                         emb_layer=emb_layer, emb_layer_p=emb_layer_p, 
                         local=local, label_mat=label_mat, **kwargs)
        self.this_turn_fake_labeled_count = [10 for _ in range(self.class_num)]
        
    def update_fake_label_count(self, y):
        self.this_turn_fake_labeled_count[y] +=1
        
    def clear_fake_label_count(self):
        # print('last clear fake label count is ', self.this_turn_fake_labeled_count)
        self.this_turn_fake_labeled_count = [10 for _ in range(self.class_num)]

    def sample_class_by_weight(self, y:int, **kwargs):
        """_summary_

        Args:
            y (_type_): positive sample index
        """
        sampled_weight_list = []
        for i in range(len(self.this_turn_fake_labeled_count)):
            if i == y:
                continue
            count = self.this_turn_fake_labeled_count[i]
            for j in range(count):
                sampled_weight_list.append(i)
        
        if len(sampled_weight_list) == 0:
            sampled_weight_list = [i for i in range(self.class_num) if i != y]
        
        sampled_y = random.sample(sampled_weight_list, 1)[0]
        return sampled_y

    def batch_unlabeled_train(self, batch, data_type: str):
        '''
        common training
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        batch_data_idx_list = result_dict['index']
        batch_data_idx_list = [i.item() for i in batch_data_idx_list]
        y = result_dict['y_data']
        
        print(dist.get_rank(), 'start unlabel train')
        
        assert batch_x[0].shape[0] == self.class_num

        total_batch_x = []
        for i in range(len(batch_x)):
            total_batch_x.append(batch_x[i])
        # print('check x and true label shape', x.shape, true_label.shape)
        total_batch_x = torch.cat(total_batch_x, dim=0)
        result_dict = self.model(total_batch_x.to(device))
        
        prob = result_dict['prob'].squeeze(1).unsqueeze(0)
        prob = prob.reshape(y.shape[0], -1)
        prob = F.softmax(prob, dim=1)
        # fake_class_list = torch.argmax(prob, dim=1)
        fake_classes_list = prob.topk(3, dim=1)[1]

        
        gap_prob = 0.5
        smallest_prob = 0.9
        choosed_text = []
        choosed_x = []
        choosed_y = []
        self.logger.info('Device %d, start find threshold for each label data train' % dist.get_rank())

        while smallest_prob >= -0.01:
            for i in range(len(batch_x)):
                # 如果已经存进fake label，则跳过
                if self.find_fake_label(batch_data_idx_list[i], data_type=data_type) != -1:
                    continue
                fake_labeles = [int(i.item()) for i in list(fake_classes_list[i])]
                # 判断是否存入fake label
                if prob[i, fake_labeles[0]] > self.pseudo_label_threshold or (prob[i, fake_labeles[0]] - prob[i, fake_labeles[1]]) > gap_prob:
                    # print('storing a fake label is', fake_labeles[0], 'true label is', y[i].item(), prob[i])
                    self.store_fake_label(fake_label_index=batch_data_idx_list[i],
                                          fake_label_id=fake_labeles[0], 
                                          true_y_id=y[i].item(),
                                          data_type=data_type)
                else:
                    if prob[i, fake_labeles[0]] >= smallest_prob:
                        texts = [text_list[j][i] for j in range(len(text_list))]
                        for fake_label in fake_labeles[:1]:
                            self.update_fake_label_count(fake_label)
                            if dist.get_rank() == 0 and self.first_batch:
                                # print('fake label is', fake_labeles, 'true label is', y[i].item(), prob[i])
                                choosed_text.append(texts[fake_label].replace('[SEP]', self.model_config.split_str))
                                choosed_x.append(batch_x[i][fake_label, :].unsqueeze(0))
                                choosed_y.append(fake_label)
            
            if len(choosed_text) > 0:
                break
            else:
                smallest_prob -= 0.05
        
        if len(choosed_text) == 0:
            self.logger.info('jump')
            generator_loss = torch.tensor(0)
        else:
            self.logger.info('Device %d, start train generator' % dist.get_rank())
            choosed_x = torch.cat(choosed_x, dim=0)
            generator_result_dict = self.generator(choosed_x.to(device),
                                                   sentence = choosed_text,
                                                   only_encoder=False)
            
            # store to disperse list
            generate_indice = generator_result_dict['quantizer_out_p']['encoding_indices']
            self.disperse_memory.update(generate_indice, choosed_y)
            
            # get generator loss
            generator_loss = generator_result_dict['loss']

        if self.model_config.detect:
            with torch.autograd.detect_anomaly():
                # backward
                print('generate loss is', generator_loss)
                generator_loss.backward()
            if self.first_batch:
                for name, param in self.generator.named_parameters():
                    if param.grad is None:
                        print('generator', name)
        else:
            generator_loss.backward()
            
        self.logger.info('Device %d, successfully backward' % dist.get_rank())
        if self.model_config.grad_norm:
            clip_grad_norm_(self.generator.parameters(),
                            self.model_config.grad_norm)

        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()
        self.first_batch = False
        if self.model_config.use_scheduler:
            self.generator_scheduler.step()
        self.logger.info('Device %d, successfully zero grad' % dist.get_rank())

        return generator_loss 

    def batch_generated_train(self, batch, final_probabs, final_targets):
        """
        输入一个generate batch，输出对所有vq结合起来生成的句子
        相比原方法加上disperse sample来辅助

        Args:
            batch (_type_): _description_
        """        
        # start epoch
        y_idx, y, y_text, label_name = batch
        y_idx = y_idx.to(device)
        y = y.to(device)

        for i in range(y.shape[0]):
            one_y_idx = y_idx[i].unsqueeze(0)
            label_start_text = y_text[i] + self.model_config.add_str
            if isinstance(self.generator, torch.nn.parallel.DistributedDataParallel):
                result_dict, p_idx = self.generator.module.sample_all(label_start_text, device=device, one_by_one=False)  # type: ignore
            else:
                result_dict, p_idx = self.generator.sample_all(label_start_text, device=device, one_by_one=False)
            sent_list = result_dict['sentences']
            if self.model_config.dataset_name == 'kesu':
                sent_list = [sent.replace(' ', '') for sent in sent_list]
            
            # if self.output_generate and self.first_batch:
            if one_y_idx.item() in self.seen_train:
                seen_type = 'train'
            elif one_y_idx.item() in self.seen_eval:
                seen_type = 'eval'
            else:
                seen_type = 'unseen'

            # output sentences
            # for j in range(len(sent_list)):
            #     print('epoch ', self.epoch, 'label', y_text[i], p_idx[j], '--- generate 句子', sent_list[j], '--- label seen type', seen_type)

            # train the model
            disperse_sample = self.disperse_memory.sample(index=one_y_idx.item(),
                                                          sample_number=len(sent_list))

            sent_list = [sent_list[disperse_index] for disperse_index in disperse_sample]

            # 从句子中重新生成batch
            negative_sent_list = []
            for i in range(len(sent_list)):
                # updata dataset
                self.dataset_memory.update_one_sentence(sent_list[i], one_y_idx.item())
                
                sent = sent_list[i].split(self.model_config.split_str)
                sampled_y = self.sample_class_by_weight(y=one_y_idx.item())
                negative_sent = [self.y_text_list[sampled_y], sent[1]]
                sent = '[SEP]'.join(sent)
                sent_list[i] = sent
                negative_sent = '[SEP]'.join(negative_sent)
                negative_sent_list.append(negative_sent)
                
            # update new data
            
            generate_positive_x, generate_negative_x, da_positive_x, da_negative_x = self.emb_augment_data_by_input_sentence(sent_list, negative_sent_list)
            
            total_loss = self.train_model_by_input_idx(generate_positive_x, generate_negative_x, da_positive_x, da_negative_x, contrastive_learning=False)
            total_loss.backward()
            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(),
                                self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False 

    def evaluate(self, loader, data_type:str ='eval', get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        self.generator.eval()
        eval_loss = 0
        # use for memory faked label instead count and true count
        faked_label_count = 0
        faked_label_true_count = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
                result_dict = batch
                batch_x = result_dict['origin']['idx']
                batch_data_idx_list = result_dict['index']
                batch_data_idx_list = [i.item() for i in batch_data_idx_list]
                y = result_dict['y_data']
                
                assert len(batch_x) == y.shape[0]
                assert batch_x[0].shape[0] == self.class_num
                
                total_true_label = []
                total_batch_x = []
                for i in range(y.shape[0]):
                    true_label = [0.0 for _ in range(self.class_num)]
                    true_label[y[i].item()] = 1.0
                    total_true_label = total_true_label + true_label
                    total_batch_x.append(batch_x[i])
                # print('check x and true label shape', x.shape, true_label.shape)
                batch_x = torch.cat(total_batch_x, dim=0)
                true_label = torch.tensor(total_true_label).unsqueeze(1)
                result_dict = self.model(batch_x.to(device),
                                         y = true_label.to(device))
                prob = result_dict['prob'].squeeze(1).unsqueeze(0)
                prob = prob.reshape(y.shape[0], -1)
                # calculate loss
                eval_loss += torch.mean(result_dict['loss']).item()
                # print('output prob', prob, prob.shape, torch.tensor([y[i].item()]), torch.tensor([y[i].item()]).shape)
                origin_final_length = len(final_targets)
                final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)
                # replace fake label on it
                for i in range(len(batch_data_idx_list)):
                # 如果已经存进fake label，则跳过
                    fake_label = self.find_fake_label(batch_data_idx_list[i], data_type=data_type)
                    if fake_label == -1:
                        pass
                    else:
                        faked_label_count += 1
                        if fake_label == y[i].item():
                            faked_label_true_count += 1
                        # probabs = [0 for i in range(self.class_num)]
                        # probabs[fake_label] = 1
                        # final_probabs[origin_final_length + i] = probabs
        self.logger.info('device %d, %d eval result are instead by fake label, %d are correct, true rate %.4f' %
                         (dist.get_rank(), faked_label_count, faked_label_true_count, float(faked_label_true_count)/(faked_label_count+1e-3)))

        return final_probabs, final_targets, eval_loss


class OneModelTrainerParallel_EntailmentImbalance_Dis(OneModelTrainer_EntailmentImbalance_Dis, Parallel_Trainer_Mixin):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        dist.init_process_group('gloo')
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def reduce_all_fake_label(self): 
        """
        Use to reduce all label data from mutli-gpus
        """
        reduce_eval_tensor = torch.tensor(self.fake_labeled_data_idx_memory_eval).to(device)
        reduce_test_tensor = torch.tensor(self.fake_labeled_data_idx_memory_test).to(device)
        time.sleep(60)
        
        group = dist.new_group([i for i in range(dist.get_world_size())])
        try:
            dist.all_reduce(reduce_eval_tensor, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(reduce_test_tensor, op=dist.ReduceOp.MAX, group=group)
        except:
            time.sleep(60)
            dist.all_reduce(reduce_eval_tensor, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(reduce_test_tensor, op=dist.ReduceOp.MAX, group=group)
        
        self.fake_labeled_data_idx_memory_eval = reduce_eval_tensor.cpu().numpy().tolist()
        self.fake_labeled_data_idx_memory_test = reduce_test_tensor.cpu().numpy().tolist()
        print('device', dist.get_rank(), 'reduced memory list', self.fake_labeled_data_idx_memory_eval[:10])
    
    def train(self):
        """
        训练传入的模型
        """
        rank = dist.get_rank()
        self.logger.info(f"Start running basic DDP example on rank {rank}.")
        # self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print(f'dataset length, train: {len(self.train_loader)}, unlabeled: {len(self.unlabel_loader)}')
        best_loss = 0
        self.best_epoch = 0
        self.only_faked_label_train = False
        # start epoch
        if self.model_config.detect is True:
            torch.autograd.set_detect_anomaly(True)            

        for epoch in range(self.max_epochs):
            self.train_loader.sampler.set_epoch(epoch)   # type: ignore
            self.eval_loader.sampler.set_epoch(epoch)  # type: ignore
            self.test_loader.sampler.set_epoch(epoch)  # type: ignore
            self.unlabel_loader.sampler.set_epoch(epoch)  # type: ignore
            self.generated_loader.sampler.set_epoch(epoch)  # type: ignore

            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            if self.model_config.ablation_discriminator:
                self.discriminator_train()
            if epoch == 0:
                for i in range(1):
                    self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                    # self.generate_train()
                    # self.generate_train()
                    self.first_epoch_generate_trian()

            if self.only_faked_label_train:
                # self.generate_train()
                pass
            else:
                for i in range(1):
                    self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                    self.label_unlabeled_data(data_type='eval')
                    self.label_unlabeled_data(data_type='test')
                    self.reduce_all_fake_label()
                    test_length, eval_length = self.rebuild_dataset()
                    if test_length < 40 or eval_length < 40:
                        self.logger.info('Start epoch %d, batch %d early stop with length %d, %d' % (self.epoch, i, test_length, eval_length))
                        self.only_faked_label_train = True
                        break
                    self.generate_train()
                    self.unlabeled_train()
                    if self.model_config.is_contrastive_t:
                        self.contrastive_train()
                # rebuild dataset, remove fake labeled data
                self.recover_dataset()
            
            self.recover_dataset()
            self.reduce_all_fake_label()
            self.rebuild_fake_labeled_dataset()
            self.fake_labeled_data_train(self.eval_loader, data_type='eval')
            self.fake_labeled_data_train(self.test_loader, data_type='test')

            self.clear_fake_label_count()
            self.recover_dataset()

            self.logger.info('Device %d, Start epoch %d, evaluation' % (dist.get_rank(), self.epoch))
            output_str = "Epoch %d \n" % self.epoch
            if dist.get_rank() == 0:
                for i in range(len(self.fake_label_true)):
                    if self.fake_label_all[i] == 0:
                        precision = 0.00
                    else:
                        precision = self.fake_label_true[i]/self.fake_label_all[i]
                    output_str = output_str + "label: %d, precision %.4f \n" % (i, precision)
                self.logger.info(output_str)

            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("Device %d evaluation precision reduce result %.4f" % (dist.get_rank(), ap_1.item()))
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                self.seen_and_unseen_test(dataloader={'test':self.single_test_loader, 'eval':self.single_eval_loader}, data_type="transductive testing")

                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()
                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)

        self.logger.info('Start test on best model')
        self.seen_and_unseen_test(dataloader=[self.single_test_loader, self.single_eval_loader], data_type="test")
        self.seen_and_unseen_test(dataloader=self.single_test_loader, data_type="test")
        self.logger.info('End test on best model')
        

class OneModelTrainerParallel_EntailmentImbalanceSemi_Dis(OneModelTrainer_EntailmentImbalance_Dis, Parallel_Trainer_Mixin):
    """Semi Supervised Version

    Args:
        OneModelTrainer_EntailmentImbalance (_type_): _description_
    """
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        dist.init_process_group('nccl')
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def sample_class_by_weight(self, y:int, split: bool=True):
        """_summary_

        Args:
            y (_type_): positive sample index
        """
        sampled_weight_list = []
        if split is True:
            for i in range(len(self.this_turn_fake_labeled_count)):
                if i == y:
                    continue
                if y in self.seen_train and i in self.seen_train:
                    count = self.this_turn_fake_labeled_count[i]
                    for j in range(count):
                        sampled_weight_list.append(i)
                elif y not in self.seen_train and i not in self.seen_train:
                    count = self.this_turn_fake_labeled_count[i]
                    for j in range(count):
                        sampled_weight_list.append(i)
            
            if len(sampled_weight_list) == 0:
                if y in self.seen_train:
                    sampled_weight_list = [i for i in range(self.class_num) if (i != y and i in self.seen_train)]
                else:
                    sampled_weight_list = [i for i in range(self.class_num) if (i != y and i not in self.seen_train)]
        else:
            sampled_weight_list = []
            for i in range(len(self.this_turn_fake_labeled_count)):
                if i == y:
                    continue
                count = self.this_turn_fake_labeled_count[i]
                for j in range(count):
                    sampled_weight_list.append(i)
            
            if len(sampled_weight_list) == 0:
                sampled_weight_list = [i for i in range(self.class_num) if i != y]
        
        sampled_y = random.sample(sampled_weight_list, 1)[0]
        return sampled_y

    def reduce_all_fake_label(self):  
        reduce_eval_tensor = torch.tensor(self.fake_labeled_data_idx_memory_eval).to(device)
        reduce_test_tensor = torch.tensor(self.fake_labeled_data_idx_memory_test).to(device)
        time.sleep(60)
        
        group = dist.new_group([i for i in range(dist.get_world_size())])
        try:
            dist.all_reduce(reduce_eval_tensor, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(reduce_test_tensor, op=dist.ReduceOp.MAX, group=group)
        except:
            time.sleep(60)
            dist.all_reduce(reduce_eval_tensor, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(reduce_test_tensor, op=dist.ReduceOp.MAX, group=group)
        
        self.fake_labeled_data_idx_memory_eval = reduce_eval_tensor.cpu().numpy().tolist()
        self.fake_labeled_data_idx_memory_test = reduce_test_tensor.cpu().numpy().tolist()
        print('device', dist.get_rank(), 'reduced memory list', self.fake_labeled_data_idx_memory_eval[:10])

    def prob_mask(self, prob: torch.Tensor, y: torch.Tensor):
        """Deal with prob in semi-supervised training setting

        Args:
            prob (torch.Tensor): _description_
            y (torch.Tensor): _description_
        """
        for i in range(y.shape[0]):
            if y[i].item() in self.seen_train:
                for j in range(prob.shape[1]):
                    if j not in self.seen_train:
                        prob[i, j] = -1e9
            else:
                for j in range(prob.shape[1]):
                    if j in self.seen_train:
                        prob[i, j] = -1e9
        return prob

    def batch_unlabeled_train(self, batch, data_type: str):
        '''
        Unlabeled train, one batch
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        batch_data_idx_list = result_dict['index']
        batch_data_idx_list = [i.item() for i in batch_data_idx_list]
        y = result_dict['y_data']
        
        assert batch_x[0].shape[0] == self.class_num

        prob = self.calculate_batch_prob(batch_x)
        prob = self.prob_mask(prob=prob, y=y)
        prob = F.softmax(prob, dim=1)
        # fake_class_list = torch.argmax(prob, dim=1)
        fake_classes_list = prob.topk(3, dim=1)[1]

        
        gap_prob = 0.5
        smallest_prob = 0.9
        choosed_text = []
        choosed_x = []
        choosed_y = []
        self.logger.info('Device %d, start find threshold for each label data train' % dist.get_rank())

        while smallest_prob >= 0.01:
            for i in range(len(batch_x)):
                # 如果已经存进fake label，则跳过
                if self.find_fake_label(batch_data_idx_list[i], data_type=data_type) != -1:
                    if self.model_config.ablation_use_new_self_training is False:
                        pass
                    else:
                        continue
                    
                fake_labeles = [int(i.item()) for i in list(fake_classes_list[i])]
                # 判断是否存入fake label
                if prob[i, fake_labeles[0]] > self.pseudo_label_threshold or (prob[i, fake_labeles[0]] - prob[i, fake_labeles[1]]) > gap_prob:
                    self.store_fake_label(fake_label_index=batch_data_idx_list[i],
                                          fake_label_id=fake_labeles[0], 
                                          true_y_id=y[i].item(),
                                          data_type=data_type)
                if prob[i, fake_labeles[0]] >= smallest_prob:
                    texts = [text_list[j][i] for j in range(len(text_list))]
                    for fake_label in fake_labeles[:1]:
                        self.update_fake_label_count(fake_label)
                        choosed_text.append(texts[fake_label].replace('[SEP]', self.model_config.split_str))
                        choosed_x.append(batch_x[i][fake_label, :].unsqueeze(0))
                        choosed_y.append(fake_label)
            
            if len(choosed_text) > 0:
                break
            else:
                smallest_prob -= 0.02
        
        if len(choosed_text) == 0:
            self.logger.info('jump')
            generator_loss = torch.tensor(0)
        else:
            self.logger.info('Device %d, start train generator' % dist.get_rank())
            choosed_x = torch.cat(choosed_x, dim=0)
            generator_result_dict = self.generator(choosed_x.to(device),
                                                   sentence = choosed_text,
                                                   only_encoder=False)
            
            # store to disperse list
            generate_indice = generator_result_dict['quantizer_out_p']['encoding_indices']
            self.disperse_memory.update(generate_indice, choosed_y)
            
            # discriminator train
            choosed_index_list = []
            for i in choosed_y:
                sampled_idx = self.disperse_memory.sample_negative(index=i, sample_number=1)[0][1]
                choosed_index_list.append(sampled_idx)
            second_emb_p = self.sample_disperse_vectors(choosed_index_list).to(device).squeeze(1)
            first_emb_p = generator_result_dict['quantizer_out_p']['quantized_stack'].squeeze(1)
            
            y_idx_vector = F.one_hot(torch.tensor(choosed_y), num_classes=self.class_num).to(device)
            # print(first_emb_p.shape, second_emb_p.shape, y_idx_vector.shape)
            discriminator_inputs = torch.cat([y_idx_vector, first_emb_p, second_emb_p], dim=1)
            y = torch.ones(len(choosed_index_list)).to(device)
            if self.model_config.ablation_discriminator:
                discriminator_result = self.discriminator(
                    inputs=discriminator_inputs,
                    label = y,
                )
                # get generator loss
                generator_loss = generator_result_dict['loss'] + discriminator_result['loss']
            else:
                generator_loss = generator_result_dict['loss']

        if self.model_config.detect:
            with torch.autograd.detect_anomaly():
                # backward
                # print('generate loss is', generator_loss)
                generator_loss.backward()
            # if self.first_batch:
            #     for name, param in self.generator.named_parameters():
            #         if param.grad is None:
            #             print('generator', name)
        else:
            generator_loss.backward()
            
        self.logger.info('Device %d, successfully backward' % dist.get_rank())
        if self.model_config.grad_norm:
            clip_grad_norm_(self.generator.parameters(),
                            self.model_config.grad_norm)

        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()
        self.first_batch = False
        if self.model_config.use_scheduler:
            self.generator_scheduler.step()
        self.logger.info('Device %d, successfully zero grad' % dist.get_rank())

        return generator_loss 

    def label_unlabeled_data(self, data_type:str):
        """
        Label the unlabeled data
        """
        self.model.eval()
        self.generator.eval()
        self.discriminator.eval()
        # conbine the training data
        if data_type == 'eval':
            dataloader = self.eval_loader
        elif data_type == 'test':
            dataloader = self.test_loader
        else:
            raise KeyError

        length = len(dataloader)
        data_loader = dataloader
        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            result_dict = batch
            batch_x = result_dict['origin']['idx']
            batch_data_idx_list = result_dict['index']
            batch_data_idx_list = [i.item() for i in batch_data_idx_list]
            y = result_dict['y_data']
            
            # print(dist.get_rank(), 'start unlabel train')
            
            assert batch_x[0].shape[0] == self.class_num

            prob = self.calculate_batch_prob(batch_x)
            prob = self.prob_mask(prob=prob, y=y)
            prob = F.softmax(prob, dim=1)
            # fake_class_list = torch.argmax(prob, dim=1)
            fake_classes_list = prob.topk(3, dim=1)[1]

            
            gap_prob = 0.5
            for i in range(len(batch_x)):
                # 如果已经存进fake label，则跳过
                if self.find_fake_label(batch_data_idx_list[i], data_type=data_type) != -1:
                    continue
                fake_labeles = [int(i.item()) for i in list(fake_classes_list[i])]
                # 判断是否存入fake label
                if prob[i, fake_labeles[0]] > self.pseudo_label_threshold or (prob[i, fake_labeles[0]] - prob[i, fake_labeles[1]]) > gap_prob:
                    self.store_fake_label(fake_label_index=batch_data_idx_list[i],
                                        fake_label_id=fake_labeles[0], 
                                        true_y_id=y[i].item(),
                                        data_type=data_type)

    def evaluate(self, loader, data_type:str ='eval', get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        self.generator.eval()
        eval_loss = 0
        # use for memory faked label instead count and true count
        faked_label_count = 0
        faked_label_true_count = 0
        faked_unseen_label_count = 0
        faked_unseen_true_count = 0
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
                result_dict = batch
                batch_x = result_dict['origin']['idx']
                batch_data_idx_list = result_dict['index']
                batch_data_idx_list = [i.item() for i in batch_data_idx_list]
                y = result_dict['y_data']
                
                assert len(batch_x) == y.shape[0]
                assert batch_x[0].shape[0] == self.class_num
                
                total_true_label = []
                total_batch_x = []
                for i in range(y.shape[0]):
                    true_label = [0.0 for _ in range(self.class_num)]
                    true_label[y[i].item()] = 1.0
                    total_true_label = total_true_label + true_label
                    total_batch_x.append(batch_x[i])
                # print('check x and true label shape', x.shape, true_label.shape)
                # batch_x = torch.cat(total_batch_x, dim=0)
                prob = self.calculate_batch_prob(total_batch_x)

                prob = self.prob_mask(prob=prob, y=y)
                prob = F.softmax(prob, dim=1)

                # print('output prob', prob, prob.shape, torch.tensor([y[i].item()]), torch.tensor([y[i].item()]).shape)
                origin_final_length = len(final_targets)
                final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)
                # replace fake label on it
                for i in range(len(batch_data_idx_list)):
                # 如果已经存进fake label，则跳过
                    fake_label = self.find_fake_label(batch_data_idx_list[i], data_type=data_type)
                    if fake_label == -1:
                        pass
                    else:
                        faked_label_count += 1
                        if fake_label == y[i].item():
                            faked_label_true_count += 1
                        if y[i].item() not in self.seen_train:
                            faked_unseen_label_count += 1
                            if fake_label == y[i].item():
                                faked_unseen_true_count += 1
                        # probabs = [0 for i in range(self.class_num)]
                        # probabs[fake_label] = 1
                        # final_probabs[origin_final_length + i] = probabs
        self.logger.info('device %d, %d eval result are instead by fake label, %d are correct, true rate %.4f, in which %d are unseen, %d is true, true rate %.4f, ' %
                         (dist.get_rank(), faked_label_count, faked_label_true_count, float(faked_label_true_count)/(faked_label_count+1e-3), faked_unseen_label_count, faked_unseen_true_count, float(faked_unseen_true_count)/(faked_unseen_label_count+1e-3)))

        return final_probabs, final_targets, eval_loss

    def train(self):
        """
        The main train flow
        """
        rank = dist.get_rank()
        self.logger.info(f"Start running basic DDP example on rank {rank}.")
        # self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print(f'dataset length, train: {len(self.train_loader)}, unlabeled: {len(self.unlabel_loader)}')
        best_loss = 0
        self.best_epoch = 0
        self.only_faked_label_train = False
        # start epoch
        if self.model_config.detect is True:
            torch.autograd.set_detect_anomaly(True)            

        for epoch in range(self.max_epochs):
            self.train_loader.sampler.set_epoch(epoch)   # type: ignore
            self.eval_loader.sampler.set_epoch(epoch)  # type: ignore
            self.test_loader.sampler.set_epoch(epoch)  # type: ignore
            self.unlabel_loader.sampler.set_epoch(epoch)  # type: ignore
            self.generated_loader.sampler.set_epoch(epoch)  # type: ignore

            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)
            if self.model_config.ablation_discriminator:
                self.discriminator_train()
            if epoch == 0:
                for i in range(1):
                    self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                    if self.model_config.ablation_generate_train is False:
                        pass
                    else:
                        self.first_epoch_generate_trian()
                        self.generate_train()
                    self.labeled_train()

            if self.only_faked_label_train:
                # self.generate_train()
                pass
            else:
                for i in range(1):
                    self.logger.info('Start epoch %d, batch %d' % (self.epoch, i))
                    self.label_unlabeled_data(data_type='eval')
                    self.label_unlabeled_data(data_type='test')
                    self.reduce_all_fake_label()
                    test_length, eval_length = self.rebuild_dataset()
                    if test_length < 100 or eval_length < 100:
                        self.logger.info('Start epoch %d, batch %d early stop with length %d, %d' % (self.epoch, i, test_length, eval_length))
                        self.only_faked_label_train = True
                        break
                    self.labeled_train()
                    if self.model_config.ablation_generate_train is False:
                        pass
                    else:
                        self.generate_train()
                    self.unlabeled_train()
                    if self.model_config.is_contrastive_t:
                        self.contrastive_train()
                # rebuild dataset, remove fake labeled data
                self.recover_dataset()
            
            self.recover_dataset()
            self.reduce_all_fake_label()
            self.rebuild_fake_labeled_dataset()
            if self.model_config.ablation_use_pseudo_label is False:
                pass
            else:
                self.fake_labeled_data_train(self.eval_loader, data_type='eval')
                self.fake_labeled_data_train(self.test_loader, data_type='test')

            if self.model_config.ablation_generate_train is True:
                self.dataset_memory.update_dataset()
                fake_dataloader = self.dataset_memory.get_dataloader(epoch=self.epoch)
                self.generated_fake_data_train(fake_dataloader)

            self.clear_fake_label_count()
            self.recover_dataset()
            
            self.disperse_memory.save_memory(epoch = self.epoch)
            self.logger.info('Device %d, Start epoch %d, evaluation' % (dist.get_rank(), self.epoch))
            output_str = "Epoch %d \n" % self.epoch
            if dist.get_rank() == 0:
                for i in range(len(self.fake_label_true)):
                    if self.fake_label_all[i] == 0:
                        precision = 0.00
                    else:
                        precision = self.fake_label_true[i]/self.fake_label_all[i]
                    output_str = output_str + "label: %d, precision %.4f \n" % (i, precision)
                self.logger.info(output_str)

            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("Device %d evaluation precision reduce result %.4f" % (dist.get_rank(), ap_1.item()))
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                self.seen_and_unseen_test(dataloader={'test':self.single_test_loader, 'eval':self.single_eval_loader}, data_type="transductive testing")

                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
        
        self.logger.info('Start test on best model')
        self.seen_and_unseen_test(dataloader=[self.single_test_loader, self.single_eval_loader], data_type="test")
        self.seen_and_unseen_test(dataloader=self.single_test_loader, data_type="test")
        self.logger.info('End test on best model')
        