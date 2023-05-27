"""
用来训练单个vq vae+GAN的架构
"""

import torch
import os
import numpy as np
import random
from tqdm import tqdm
from codes.utils import save_data, MemTracker
from itertools import cycle


from torch.nn.utils.clip_grad import clip_grad_norm_
from model.train_base import Trainbasic
from model.new_model.model.model import Fix_Model

from model.new_model.utils.loss import ContrastiveLoss, ContrastiveLoss_Neg

from multiprocessing import cpu_count
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed as data_dist
import transformers

world_size = 8
cpu_num = max(1, int(cpu_count()/world_size) - 1)  # 自动获取最大核心数目
print('cpu num is', cpu_num)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class NewModelTrainerBasic(Trainbasic):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset,
                 emb_layer=None, emb_layer_p=None, local=False, **kwargs):

        random.seed(114514)
        os.environ['PYTHONHASHSEED'] = str(114514)
        np.random.seed(114514)
        torch.manual_seed(114514)
        torch.cuda.manual_seed(114514)
        torch.cuda.manual_seed_all(114514)
        super().__init__(model_config, logger, emb_layer=emb_layer, local=local, **kwargs)
        # help tracker
        self.gpu_tracker = MemTracker()
        # 训练组件设置
        self.output_generate = model_config.output_generate
        self.unlabel = model_config.is_unlabel_train  # 是否使用unlabel dataset训练
        self.generate = model_config.is_generate_train # 是否使用生成的数据训练
        # 学习参数
        self.max_epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.generate_size = model_config.generate_size
        self.div_batch_size = model_config.batch_size
        self.lr = model_config.lr
        self.class_num = model_config.class_num
        self.class_loss_decay = model_config.class_loss_decay
        self.class_loss_decay_weight = 1
        # embedding layer p if bert
        self.emb_layer_p = emb_layer_p
        # save path
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        # pseudo label 参数
        self.pseudo_label_threshold = model_config.pseudo_label_threshold
        # load dataset
        print('load dataset')
        self.load_dataset(dataloader_list=dataloader_list, generated_dataset=generated_dataset)
        # 分seen和unseen clas
        print('split seen and unseen id')
        self.get_seen_class_id()
        print('seen id list is: ', self.seen_train, 'eval id list is:', self.seen_eval, 'test id list is:', self.seen_test)
        
        try:
            if label_mat.shape[1] == 1:
                label_mat = label_mat.squeeze(1)
            self.label_mat = label_mat.to(device)
        except Exception as e:
            print(e, label_mat, 'label matrix broken')
            self.label_mat = label_mat.to(device)
            raise RuntimeError
        
        # 记住之前的distil和disperse组合
        self.pair_memory = {}
        for i in range(self.class_num):
            self.pair_memory[i] = {}

    def update_class_loss_weight(self, max_epoch = 10):
        if self.class_loss_decay and self.epoch <= max_epoch:
            self.class_loss_decay_weight = 1 - 0.99 * self.epoch/max_epoch
        print('new class loss weight is %.4f' % self.class_loss_decay_weight)

    def batch_gen_train(self, batch, final_targets, final_probabs, is_train=True):
        '''
        common training
        '''
        # return loss
        total_loss = torch.tensor(0)
        loss_dict = {
            'contrastive_t_loss': 0,
            'contrastive_p_loss': 0,
            'recon loss': 0,
            'vq_loss': 0,
            'classifier_loss': 0,
            'discriminator_loss': 0
            
        }

        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx'].to(device)
        batch_x_text = result_dict['origin']['text']
        batch_da_x = result_dict['da']['idx'].to(device)
        batch_da_x_text = result_dict['origin']['text']
        
        y = result_dict['y_data']

        # get result
        result_dict = self.model(batch_x,
                                 y=y.to(device),
                                 sentence=batch_x_text
                                 )
        # get da result
        da_result_dict = self.model(batch_da_x,
                                    y=y.to(device),
                                    sentence=batch_da_x_text)

        # calculate probability
        class_prob = result_dict['classifier']['prob']
        final_targets, final_probabs = self.get_prob_and_target(y, class_prob, final_targets, final_probabs)

        if is_train:
            # 检测生成的句子
            if self.output_generate and self.first_batch:
                sent_list = self.decoder_idx_to_sentence(result_dict=result_dict['decoder_output'], need_build=False)
                for i in range(len(sent_list)):
                    print('epoch:', self.epoch, '--- recon句子', sent_list[i], '--- origin sentence 句子', text_list[i],
                          '--- p idx', result_dict['quantizer_out_p']['encoding_indices'][i],'---t idx')
                    # store to output some visible result
                    p_idx = result_dict['quantizer_out_p']['encoding_indices'][i].item()
                    y_idx = y[i].item()
                    if p_idx not in self.pair_memory[y_idx].keys():
                        self.pair_memory[y_idx][p_idx] = [{'recon': sent_list[i], 'origin':text_list[i]}]
                    else:
                        self.pair_memory[y_idx][p_idx].append({'recon': sent_list[i], 'origin':text_list[i]})

            # start deal with loss
            # reconstruction loss
            total_loss = result_dict['loss'] + da_result_dict['loss'] # recon loss + vq t loss + vq p loss
            total_loss = torch.sum(total_loss)/(self.div_batch_size * 2)
            loss_dict['recon loss'] = result_dict['decoder_output']['loss_reconstruct'].item()/self.div_batch_size
            loss_dict['vq_loss'] = total_loss.item() - loss_dict['recon loss']
            
            # calculate contrastive loss
            if self.model_config.is_contrastive_t:
                z_t = result_dict['z_t'].squeeze(1)
                z_t_da = da_result_dict['z_t'].squeeze(1)
                # print(z_t.shape, z_t_da.shape)
                contrastive_loss_t = self.contras_loss_fun(z_t, z_t_da) * self.model_config.contrastive_t_coef
                total_loss = total_loss + contrastive_loss_t
                loss_dict['contrastive_t_loss'] = loss_dict['contrastive_t_loss'] + contrastive_loss_t.item()
            if self.model_config.is_contrastive_p:
                p_result_dict = self.get_contrastive_p_loss(result_dict, da_result_dict)
                total_loss = total_loss + p_result_dict['contrastive_loss_p']  * self.model_config.contrastive_p_coef
                loss_dict['contrastive_p_loss'] = loss_dict['contrastive_p_loss'] + p_result_dict['contrastive_loss_p'].item()
                
            # 分类loss
            # classifier loss
            loss_dict['classifier_loss'] = loss_dict['classifier_loss'] + result_dict['classifier']['loss'].item()
            total_loss = result_dict['classifier']['loss'] + da_result_dict['classifier']['loss'] + (total_loss * self.class_loss_decay_weight)

        return loss_dict, final_targets, final_probabs, total_loss

    def unlabel_gen_train(self, batch):
        # deal batch data
        result_dict = batch
        batch_x = result_dict['origin']['idx'].to(device)
        batch_x_text = result_dict['origin']['text']
        batch_da_x = result_dict['da']['idx'].to(device)
        batch_da_x_text = result_dict['origin']['text'] 

        # get result
        result_dict = self.model(batch_x, sentence=batch_x_text)
        # get da result
        da_result_dict = self.model(batch_da_x, sentence=batch_da_x_text)

        # start deal with loss
        # reconstruction loss
        total_loss = result_dict['loss'] + da_result_dict['loss'] # recon loss + vq t loss + vq p loss
        total_loss = total_loss + (result_dict['classifier']['loss'] + da_result_dict['classifier']['loss']) * 0.001
        total_loss = torch.sum(total_loss)/(self.div_batch_size * 2)
        
        # calculate contrastive loss
        if self.model_config.is_contrastive_t:
            z_t = result_dict['z_t'].squeeze(1)
            z_t_da = da_result_dict['z_t'].squeeze(1)
            print(z_t.shape, z_t_da.shape)
            total_loss = total_loss + self.contras_loss_fun(z_t, z_t_da) * self.model_config.contrastive_t_coef
        if self.model_config.is_contrastive_p:
            p_result_dict = self.get_contrastive_p_loss(result_dict, da_result_dict)
            total_loss = total_loss + p_result_dict['contrastive_loss_p'] * self.model_config.contrastive_p_coef

        
        return total_loss * self.class_loss_decay_weight

    def generate_gen_train(self, batch):
        """生成数据训练分类器

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """        
        y_idx, y, y_text = batch
        y_idx = y_idx.to(device)
        y = y.to(device)

        with torch.no_grad():
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                result_dict, y_idx, p_idx = self.model.module.sample_result(y_idx, device=device, label_text=y_text)  # type: ignore
            else:
                result_dict, y_idx, p_idx = self.model.sample_result(y_idx, device=device, label_text=y_text)
        # 生成句子
        sent_list = self.decoder_idx_to_sentence(result_dict=result_dict, need_build=False)

        if self.output_generate and self.first_batch:
            for i in range(len(sent_list)):
                print('epoch:', self.epoch, 'generate 句子', sent_list[i], 'label', y_text[i], p_idx[i])
        # 去掉空句子，防止nan生成
        
        # 从句子中重新生成batch
        batch_x, x_pad_mask = self.emb_data(sent_list)
        # get result
        result_dict = self.model(batch_x, y=y_idx.to(batch_x.device), sentence=sent_list)
        print('gen result dict', result_dict)

        total_loss = result_dict['loss']
        total_loss = torch.sum(total_loss)/(self.div_generate_size * 2)
        
        # classifier loss
        total_loss = total_loss * self.class_loss_decay_weight + result_dict['classifier']['loss']
        return total_loss
    
    def decoder_idx_to_sentence(self, result_dict, need_build=True):
        if self.model_config.decoder_type in ['GPT2']:
            sent_list = result_dict['sentences']
        else:
            sent_list, _, _ = self.generate_sentence(result_dict['pred_idx'], need_build=False)
        return sent_list
    
    def evaluate(self, loader, **kwargs):
        self.model.eval()
        self.logger.info('Start model test with name: %s' % (self.model_path))
        final_targets = []
        final_probabs = []
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader):
                # deal batch data
                result_dict = batch
                batch_x = result_dict['origin']['idx'].to(device)
                batch_x_text = result_dict['origin']['text']
                y = result_dict['y_data']

                # get result
                result_dict = self.model(batch_x,
                                        sentence=batch_x_text
                                        )
                # calculate probability
                class_prob = result_dict['classifier']['prob']
                final_targets, final_probabs = self.get_prob_and_target(y, class_prob, final_targets, final_probabs)

        return final_probabs, final_targets, eval_loss