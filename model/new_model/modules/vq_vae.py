import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.modules.encoder import Encoder
from model.new_model.modules.decoder import Decoder
from model.new_model.utils.loss import CEFocalLoss
from model.new_model.utils.vq import VQ
import numpy as np


class VQ_VAE_Base(nn.Module):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, **kwargs):
        super().__init__()
        # global setting
        self.config = config
        self.device = device
        # gpu tracker
        self.gpu_tracker = gpu_tracker
        # size configßß
        self.k = config.class_num  # 一共有多少个类
        self.emb_size = config.emb_size  # input size
        self.distil_size = config.distil_size  # dislling latent size
        self.disper_size = config.disper_size  # dispersing latent size
        self.disper_num = config.disper_num
        # vq type config
        self.t_vq_type = config.distil_vq_type  # Soft or VQ or Fix
        self.p_vq_type = config.disperse_vq_type  # Soft or VQ or Fix
        # encoder and decode type config
        self.encoder_type = config.encoder_type  # encoder type
        try:
            self.encoder_p_type = config.encoder_p_type
        except Exception as e:
            print(e, 'not encoder p type')
            self.encoder_p_type = config.encoder_type
        self.decoder_type = config.decoder_type  # decoder type
        # loss coefficient
        self.vq_coef = config.vq_coef
        self.comit_coef = config.comit_coef

        # embeding layer
        self.emb_layer = emb_layer

    def _init_model(self, bos):
        raise NotImplementedError

    def sample_result(self, y_idx, device, n):
        """Returns embedding tensor for a batch of indices."""
        codebook_emb_t = self.emb_t.quantize_embedding(y_idx)  # [batch_size, distill_size]
        # random choose n samples from emb p
        n = y_idx.shape[0]

        p_idx = np.random.randint(0, self.disper_num, size=n)
        p_idx = torch.from_numpy(p_idx).to(device)

        codebook_emb_p = self.emb_p.quantize_embedding(p_idx)  # [batch_size, disperse_size]
        # result
        result = torch.cat([codebook_emb_t.unsqueeze(1), codebook_emb_p.unsqueeze(1)], dim=1).to(device)
        result = self.decoder(None, result, is_training=False)  # [batch_size, emb_size]
        return result.detach(), y_idx, p_idx  # 截断梯度流

    def sample_all(self, y_idx, device):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        codebook_emb_t = self.emb_t.quantize_embedding(y_idx)  # [1, distill_size]

        p_idx = torch.arange(0, self.disper_num, 1).to(device)
        
        codebook_emb_p = self.emb_p.quantize_embedding(p_idx)  # [disper num, disperse_size]
        # result
        result = torch.cat([codebook_emb_t.unsqueeze(1).repeat(codebook_emb_p.shape[0], 1, 1), codebook_emb_p.unsqueeze(1)], dim=1).to(device)
        result = self.decoder(None, result, is_training=False)  # [batch_size, emb_size]
        return result.detach(), y_idx, p_idx  # 截断梯度流


class VQ_VAE(VQ_VAE_Base):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, label_mat=None, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, **kwargs)
        self._init_model(bos, label_mat, **kwargs)

    def _init_model(self, bos, label_mat, **kwargs):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.distil_size,
                                                          self.gpu_tracker, encoder_name='distilling')
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing')
        # distilling vq embedding
        print('distilling vq type is', self.t_vq_type)
        self.emb_t = VQ[self.t_vq_type](self.k, self.disper_size, self.comit_coef, q_name='vq distilling',
                                        label_mat=label_mat)
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, q_name='common vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer,
                                                  **kwargs)

    def forward(self, x, x_pad_mask=None, **kwargs):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # 计算latent variable
        z_t, var_t = self.encoder_distill(x, pad_mask=x_pad_mask)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(x, pad_mask=x_pad_mask)  # [batch_size, emb_size]
        # 找最近的离散变量
        z_t_hat, loss_t = self.emb_t(z_t, var_t, smooth=False)  # [batch_size, emb_size]
        emb_p_dict = self.emb_p(z_p, var_p, smooth=False)  # [batch_size, emb_size]
        z_p_hat = emb_p_dict['quantized']
        loss_p = emb_p_dict['loss']
        # decoder
        z_total = torch.cat([z_t_hat, z_p_hat], dim=1)  # [batch_size, emb_size]
        result = self.decoder(x, z_total, pad_mask=x_pad_mask, is_training=True)  # [batch_size, seq_len-1, emb_size]
        result_dict = {
            'recon_result': result,
            'z_t': z_t,
            'z_p': z_p,
            'loss_t': loss_t,
            'loss_p': loss_p,
        }
        return result_dict

    def recon_loss_function(self, result_dict, x, x_pad_mask):
        recon_x = result_dict['recon_result']
        if x_pad_mask is None:
            x = x[:, 1:, :]  # [batch_size, seq_len-1, emb_size]
            recon_x = recon_x[:, 1:, :]
            ce_loss = F.mse_loss(recon_x.float(), x.float())
        else:
            mask = x_pad_mask[:, 1:]
            mask = (mask.unsqueeze(2) == 1).to(recon_x.device)
            x = torch.masked_select(x[:, 1:, :], mask)  # [batch_size, seq_len-1, emb_size]
            recon_x = torch.masked_select(recon_x[:, 1:, :], mask)
            ce_loss = F.mse_loss(recon_x.float(), x.float())
        return ce_loss

    def loss_function(self, result_dict, x, x_pad_mask, recon_loss=True, **kwargs):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, emb_size]
        '''
        loss_t = result_dict['loss_t']
        loss_p = result_dict['loss_p']
        if recon_loss is False:
            return self.vq_coef*loss_t + self.vq_coef*loss_p
        else:
            ce_loss = self.recon_loss_function(result_dict, x, x_pad_mask)
            return ce_loss + self.vq_coef*loss_t + self.vq_coef*loss_p


class Fix_VQ_VAE(VQ_VAE):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, label_mat, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, label_mat=label_mat, **kwargs)

    def forward(self, x, x_pad_mask=None, y_idx=None, **kwargs):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # 计算latent variable
        z_t, var_t = self.encoder_distill(x, pad_mask=x_pad_mask)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(x, pad_mask=x_pad_mask)  # [batch_size, emb_size]
        
        # 找最近的离散变量, output为-distance，用于计算cross entropy loss
        z_t_hat, output = self.emb_t(z_t, var_t, smooth=False)  # [batch_size, emb_size]
        # 找最近的离散变量，并且返回loss
        emb_p_dict = self.emb_p(z_p, var_p, smooth=False)  # [batch_size, emb_size]
        z_p_hat = emb_p_dict['quantized']
        loss_p = emb_p_dict['loss']

        # decoder
        if y_idx is None:
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        else:
            z_t_hat = self.emb_t.quantize_embedding(y_idx)
            z_total = torch.cat([z_t_hat.unsqueeze(1).detach(), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]
        result = self.decoder(x, z_total, pad_mask=x_pad_mask, is_training=True)  # [batch_size, seq_len-1, emb_size]
        result_dict = {
            'recon_result': result,
            'z_t': z_t,
            'z_p': z_p,
            'loss_p': loss_p,
            'prob': output
        }
        return result_dict

    def loss_function(self, result_dict, x, x_pad_mask, recon_loss=True, **kwargs):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, emb_size]
        '''
        loss_p = result_dict['loss_p']
        if recon_loss is False:
            return self.vq_coef*loss_p
        else:
            ce_loss = self.recon_loss_function(result_dict, x, x_pad_mask)
            print('ce loss is', ce_loss)
            return ce_loss + self.vq_coef*loss_p


class VQ_VAE_Idx(VQ_VAE_Base):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, label_mat=None, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, **kwargs)
        self.distil_num = self.config.distill_num
        self._init_model(bos, label_mat)

    def _init_model(self, bos, label_mat):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.distil_size,
                                                          self.gpu_tracker, encoder_name='distilling')
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing')
        # distilling vq embedding
        print('distilling vq type is', self.t_vq_type)
        self.emb_t = VQ[self.t_vq_type](self.distil_num, self.distil_size, self.comit_coef, q_name='vq distilling',
                                        label_mat=label_mat)
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, q_name='common vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer)

    def forward(self, inputs, x_pad_mask=None, **kwargs):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # encode
        z_t, var_t, z_p, var_p = self.encode(inputs=inputs, pad_mask=x_pad_mask)
        # vq
        z_t_hat, loss_t, z_p_hat, loss_p = self.vq(z_t, var_t, z_p, var_p)
        # decoder
        z_total = torch.cat([z_t_hat.unsqueeze(1), z_p_hat.unsqueeze(1)], dim=1)
        # 使用inputs复原
        # [batch_size, seq_len-1, emb_size]
        result = self.decoder(inputs, z_total, pad_mask=x_pad_mask, is_training=True)

        result_dict = {
            'recon_result': result,
            'z_t': z_t,
            'z_p': z_p,
            'loss_t': loss_t,
            'loss_p': loss_p,
        }
        return result_dict

    def encode(self, inputs, pad_mask):
        # 计算x
        x = self.emb_layer.emb_model.emb_from_idx(inputs).to(inputs.device)
        # 计算latent variable
        z_t, var_t = self.encoder_distill(x, pad_mask=pad_mask)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(x, pad_mask=pad_mask)  # [batch_size, emb_size]
        return z_t, var_t, z_p, var_p

    def vq(self, z_t, var_t, z_p, var_p, smooth=False):
        # 找最近的离散变量
        z_t_hat, loss_t = self.emb_t(z_t, var_t, smooth=smooth)  # [batch_size, emb_size]
        emb_p_dict = self.emb_p(z_p, var_p, smooth=False)  # [batch_size, emb_size]
        z_p_hat = emb_p_dict['quantized']
        loss_p = emb_p_dict['loss']
        return z_t_hat, loss_t, z_p_hat, loss_p

    def recon_loss_function(self, result_dict, x, x_pad_mask):
        recon_x = result_dict['recon_result']
        loss_fn = CEFocalLoss().to(recon_x.device)
        # loss_fn = nn.CrossEntropyLoss().to(recon_x.device)
        if x_pad_mask is None:
            x = x[:, :]  # [batch_size, seq_len-1, emb_size]
            recon_x = recon_x[:, :, :]
            final_ce_loss = loss_fn(recon_x, x)
        else:
            mask = x_pad_mask[:, :]
            x_mask = (mask == 1).to(recon_x.device)
            # [batch_size, seq_len-1, emb_size]
            x = torch.masked_select(x[:, :].to(recon_x.device), x_mask)
            recon_prob_mask = (mask.unsqueeze(2) == 1).to(recon_x.device)
            masked_recon_x = torch.masked_select(recon_x[:, :, :], recon_prob_mask)
            masked_recon_x = masked_recon_x.reshape(-1, recon_x.shape[2])
            final_ce_loss = loss_fn(masked_recon_x, x)
        return final_ce_loss

    def loss_function(self, result_dict, x, x_pad_mask, recon_loss=True, **kwargs):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, emb_size]
        '''
        loss_t = result_dict['loss_t']
        loss_p = result_dict['loss_p']
        if recon_loss is False:
            return self.vq_coef*loss_t + self.vq_coef*loss_p
        else:
            ce_loss = ce_loss = self.recon_loss_function(result_dict, x, x_pad_mask)
            print('recon loss is', ce_loss)
            return ce_loss + self.vq_coef*loss_t + self.vq_coef*loss_p


class Fix_VQ_VAE_Idx(VQ_VAE_Idx):
    def __init__(self, config, device, gpu_tracker, bos, label_mat, emb_layer, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, label_mat=label_mat, **kwargs)

    def forward(self, inputs, x_pad_mask=None, easy=True, pretrain=False, y_idx=None, **kwargs):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        if pretrain is False:
            # encode
            z_t, var_t, z_p, var_p = self.encode(inputs=inputs, pad_mask=x_pad_mask)

            # 找最近的离散变量, output为-distance，用于计算cross entropy loss
            z_t_hat, output, z_p_hat, loss_p = self.vq(z_t, var_t, z_p, var_p)

            # decoder
            # z_t_hat = z_t_hat.unsqueeze(1).detach()
            z_t_hat = z_t_hat.detach()
            z_total = torch.cat([z_t_hat.unsqueeze(1), z_p_hat.unsqueeze(1)], dim=1)  # [batch_size, 2, emb_size]

            # [batch_size, seq_len-1, emb_size]
            result = self.decoder(inputs, z_total, pad_mask=x_pad_mask, is_training=True, easy=easy,
                                  pretrain=pretrain)
            result_dict = {
                'recon_result': result,
                'z_t': z_t,
                'z_p': z_p,
                'loss_p': loss_p,
                'prob': output
            }
            return result_dict
        else:
            result = self.decoder(inputs, None, pad_mask=x_pad_mask, is_training=True, easy=easy,
                                  pretrain=pretrain)
            result_dict = {
                'recon_result': result,
            }
        return result_dict

    def loss_function(self, result_dict, x, x_pad_mask, recon_loss=True, vq_loss=True, **kwargs):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, ntokens]
        '''
        if vq_loss is True:
            loss_p = result_dict['loss_p']
            final_vq_loss = self.vq_coef * loss_p
        if recon_loss is True:
            final_ce_loss = self.recon_loss_function(result_dict, x, x_pad_mask)
        if vq_loss is True and recon_loss is True:
            return final_vq_loss + final_ce_loss
        elif vq_loss is True and recon_loss is False:
            return final_vq_loss
        elif vq_loss is False and recon_loss is True:
            return final_ce_loss
        else:
            return None


VAE = {
    'VQ_VAE': VQ_VAE,
    'Fix_VQ_VAE': Fix_VQ_VAE,
    'VQ_VAE_Idx': VQ_VAE_Idx,
    'Fix_VQ_VAE_Idx': Fix_VQ_VAE_Idx
}
