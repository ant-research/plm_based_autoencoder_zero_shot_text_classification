import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.model.vq_vae import VQ_VAE_Base
from model.new_model.bert_model.bert_encoder import Encoder
from model.new_model.bert_model.bert_decoder import Decoder
from model.new_model.utils.vq import VQ, FixedVectorQuantizer
import numpy as np


class VQ_VAE_Idx_BERT(VQ_VAE_Base):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer_t, emb_layer_p, label_mat=None):
        super().__init__(config, device, gpu_tracker, bos, emb_layer_t)
        # emb_layer
        self.emb_layer_t = emb_layer_t
        self.emb_layer_p = emb_layer_p
        self._init_model(bos, label_mat)

    def _init_model(self, bos, label_mat):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.distil_size,
                                                          self.gpu_tracker, encoder_name='distilling',
                                                          emb_layer=self.emb_layer_t)
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing',
                                                             emb_layer=self.emb_layer_p)
        # distilling vq embedding
        print('distilling vq type is', self.t_vq_type)
        self.emb_t = VQ[self.t_vq_type](self.k, self.disper_size, self.comit_coef, q_name='vq distilling',
                                        label_mat=label_mat)
        # dispersing vq embedding
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, q_name='common vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer_t)

    def forward(self, inputs_t, inputs_p, x_pad_mask=None, x_pad_mask_p=None):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # encode
        z_t, var_t, z_p, var_p = self.encode(inputs_t=inputs_t, inputs_p=inputs_p,
                                             pad_mask=x_pad_mask, pad_mask_p=x_pad_mask_p)
        # vq
        z_t_hat, loss_t, z_p_hat, loss_p = self.vq(z_t, var_t, z_p, var_p)
        # decoder
        z_total = torch.cat([z_t_hat, z_p_hat], dim=1)  # [batch_size, emb_size]
        # 使用inputs复原
        result = self.decoder(inputs_t, z_total, pad_mask=x_pad_mask, is_training=True)  # [batch_size, seq_len-1, emb_size]
        return result, z_t_hat, z_t, z_p_hat, z_p, loss_t, loss_p

    def encode(self, inputs_t, inputs_p, pad_mask, pad_mask_p):
        # 计算latent variable
        z_t, var_t = self.encoder_distill(inputs_t, pad_mask=pad_mask, detach=True)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(inputs_p, pad_mask=pad_mask_p, detach=False)  # [batch_size, emb_size]
        return z_t, var_t, z_p, var_p

    def vq(self, z_t, var_t, z_p, var_p, smooth=False):
        # 找最近的离散变量
        z_t_hat, loss_t = self.emb_t(z_t, var_t, smooth=smooth)  # [batch_size, emb_size]
        z_p_hat, loss_p = self.emb_p(z_p, var_p, smooth=smooth)  # [batch_size, emb_size]
        return z_t_hat, loss_t, z_p_hat, loss_p

    def loss_function(self, x, recon_x, loss_t, loss_p, x_pad_mask, generate=False):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, emb_size]
        '''
        if generate:
            return self.vq_coef*loss_t + self.vq_coef*loss_p
        else:
            if x_pad_mask is None:
                x = x[:, :, :]  # [batch_size, seq_len-1, emb_size]
                recon_x = recon_x[:, :, :]
                recon_loss = F.mse_loss(recon_x.float(), x.float())
            else:
                mask = x_pad_mask[:, :]
                mask = (mask.unsqueeze(2) == 1).to(recon_x.device)
                x = torch.masked_select(x[:, :, :], mask)  # [batch_size, seq_len-1, emb_size]
                recon_x = torch.masked_select(recon_x[:, :, :], mask)
                recon_loss = F.mse_loss(recon_x.float(), x.float())
            print('recon loss is', recon_loss)
            return recon_loss + self.vq_coef*loss_t + self.vq_coef*loss_p


class Fix_VQ_VAE_Idx_BERT(VQ_VAE_Idx_BERT):
    def __init__(self, config, device, gpu_tracker, bos, label_mat, emb_layer_t, emb_layer_p):
        super().__init__(config, device, gpu_tracker, bos, emb_layer_t=emb_layer_t, emb_layer_p=emb_layer_p,
                         label_mat=label_mat)

    def forward(self, inputs_t, inputs_p, x_pad_mask=None, x_pad_mask_p=None, easy=True, y_idx=None):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # encode
        z_t, var_t, z_p, var_p = self.encode(inputs_t=inputs_t, inputs_p=inputs_p,
                                             pad_mask=x_pad_mask, pad_mask_p=x_pad_mask_p)

        # 找最近的离散变量, output为-distance，用于计算cross entropy loss
        z_t_hat, output, z_p_hat, loss_p = self.vq(z_t, var_t, z_p, var_p)

        # decoder
        if y_idx is None:
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        else:
            z_t_hat = self.emb_t.quantize_embedding(y_idx)
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        # [batch_size, seq_len-1, emb_size]
        result = self.decoder(inputs_t, z_total, pad_mask=x_pad_mask, is_training=True, easy=easy)
        return result, z_t, z_p, loss_p, output

    def loss_function(self, inputs, recon_x_prob, loss_p, x_pad_mask, generate=False):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, ntokens]
        '''
        if generate:
            return self.vq_coef*loss_p
        else:
            loss_fn = nn.CrossEntropyLoss().to(recon_x_prob.device)
            if x_pad_mask is None:
                x = inputs[:, :]  # [batch_size, seq_len-1, emb_size]
                recon_x = recon_x_prob[:, :, :]
                ce_loss = loss_fn(recon_x, x)
            else:
                print('vq vae loss output', recon_x_prob.shape, inputs.shape)
                mask = x_pad_mask[:, :]
                x_mask = (mask == 1).to(recon_x_prob.device)
                x = torch.masked_select(inputs[:, :].to(recon_x_prob.device), x_mask)  # [batch_size, seq_len-1, emb_size]
                recon_prob_mask = (mask.unsqueeze(2) == 1).to(recon_x_prob.device)
                recon_x = torch.masked_select(recon_x_prob[:, :, :], recon_prob_mask)
                recon_x = recon_x.reshape(-1, recon_x_prob.shape[2])
                ce_loss = loss_fn(recon_x, x)
            print('ce loss is', ce_loss)
            return ce_loss + self.vq_coef*loss_p


class Fix_VQ_VAE_Idx_BERTGCN(VQ_VAE_Idx_BERT):
    def __init__(self, config, device, gpu_tracker, bos, label_mat, emb_layer_t, emb_layer_p, parent_adj, child_adj):
        self.parent_adj = parent_adj
        self.child_adj = child_adj
        super().__init__(config, device, gpu_tracker, bos, emb_layer_t=emb_layer_t, emb_layer_p=emb_layer_p,
                         label_mat=label_mat)

    def _init_model(self, bos, label_mat):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.distil_size,
                                                          self.gpu_tracker, encoder_name='distilling',
                                                          emb_layer=self.emb_layer_t)
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing',
                                                             emb_layer=self.emb_layer_p)
        # distilling vq embedding
        print('distilling vq type is', self.t_vq_type)
        self.emb_t = VQ[self.t_vq_type](self.k, self.disper_size, self.comit_coef, q_name='vq distilling',
                                        label_mat=label_mat, adj_parent=self.parent_adj.to(label_mat.device),
                                        adj_child=self.child_adj.to(label_mat.device))
        # dispersing vq embedding
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, q_name='common vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer_t)

    def forward(self, inputs_t, inputs_p, x_pad_mask=None, x_pad_mask_p=None, easy=True, y_idx=None):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # encode
        z_t, var_t, z_p, var_p = self.encode(inputs_t=inputs_t, inputs_p=inputs_p,
                                             pad_mask=x_pad_mask, pad_mask_p=x_pad_mask_p)

        # 找最近的离散变量, output为-distance，用于计算cross entropy loss
        z_t_hat, output, z_p_hat, loss_p = self.vq(z_t, var_t, z_p, var_p)

        # decoder
        if y_idx is None:
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        else:
            z_t_hat = self.emb_t.quantize_embedding(y_idx)
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        # [batch_size, seq_len-1, emb_size]
        result = self.decoder(inputs_t, z_total, pad_mask=x_pad_mask, is_training=True, easy=easy)
        return result, z_t, z_p, loss_p, output

    def loss_function(self, inputs, recon_x_prob, loss_p, x_pad_mask, generate=False):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, ntokens]
        '''
        if generate:
            return self.vq_coef*loss_p
        else:
            loss_fn = nn.CrossEntropyLoss().to(recon_x_prob.device)
            if x_pad_mask is None:
                x = inputs[:, :]  # [batch_size, seq_len-1, emb_size]
                recon_x = recon_x_prob[:, :, :]
                ce_loss = loss_fn(recon_x, x)
            else:
                print('vq vae loss output', recon_x_prob.shape, inputs.shape)
                mask = x_pad_mask[:, :]
                x_mask = (mask == 1).to(recon_x_prob.device)
                x = torch.masked_select(inputs[:, :].to(recon_x_prob.device), x_mask)  # [batch_size, seq_len-1, emb_size]
                recon_prob_mask = (mask.unsqueeze(2) == 1).to(recon_x_prob.device)
                recon_x = torch.masked_select(recon_x_prob[:, :, :], recon_prob_mask)
                recon_x = recon_x.reshape(-1, recon_x_prob.shape[2])
                ce_loss = loss_fn(recon_x, x)
            print('ce loss is', ce_loss)
            return ce_loss + self.vq_coef*loss_p


VAE = {
    'VQ_VAE': VQ_VAE_Idx_BERT,
    'Fix_VQ_VAE': Fix_VQ_VAE_Idx_BERT,
    'Fix_GCN': Fix_VQ_VAE_Idx_BERTGCN
}
