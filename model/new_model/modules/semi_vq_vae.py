"""
Semi VQ_VAE model：
disperse部分：VQ-VAE
distill部分：AE
classifier：Euclidean distance
discriminator：D
loss：
1. disperse vq loss
2. reconstruction loss
3. classifier loss
4. discriminator loss
5. contrastive loss t
6. contrastive loss p
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.modules.encoder import Encoder
from model.new_model.modules.decoder import Decoder
from model.new_model.utils.loss import CEFocalLoss
from model.new_model.utils.vq import VQ


class VQ_VAE_Base(nn.Module):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, **kwargs):
        super().__init__()
        # global setting
        self.config = config
        self.device = device

        # gpu tracker
        self.gpu_tracker = gpu_tracker

        # size config
        self.k = config.class_num  # 一共有多少个类
        self.emb_size = config.emb_size  # input size
        self.disper_size = config.disper_size  # dispersing latent size
        self.disper_num = config.disper_num
        self.decompose_number = config.decompose_number

        # vq type config
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

    def forward(self, x, x_pad_mask=None, only_encoder=False, use_vq=True, **kwargs):
        '''
        x: [batch_size, seq_len, emb_size]
        '''
        # encode
        z_t, var_t, z_p, var_p = self.encode(inputs=x, pad_mask=x_pad_mask)
        # vq
        z_t_hat = z_t  # [batch_size, emb_size]
        if only_encoder is True:
            result_dict = {
                'z_t': z_t,
            }
            return result_dict
        else:
            # use clustering center as vq
            if use_vq is False:
                print('z_p is', z_p)
                z_p_hat = z_p
                loss_p = torch.tensor(0).to(z_p.device)
                encoding_indices = torch.zeros(z_p.shape[0]).to(z_p.device)
            else:
                emb_p_dict = self.emb_p(z_p, var_p, smooth=False)  # [batch_size, emb_size]
                z_p_hat = emb_p_dict['quantized']
                loss_p = emb_p_dict['loss']
                encoding_indices = emb_p_dict['encoding_indices']
            # decoder
            if self.p_vq_type == 'DVQ':
                z_total = torch.cat([z_t_hat.unsqueeze(1), z_p_hat], dim=1)
            else:
                z_total = torch.cat([z_t_hat.unsqueeze(1), z_p_hat.unsqueeze(1)], dim=1)
            # 使用inputs复原
            # [batch_size, seq_len-1, emb_size]
            result = self.decoder(x, z_total, pad_mask=x_pad_mask, is_training=True)

            result_dict = {
                'recon_result': result,
                'z_t': z_t,
                'z_p': z_p,
                'loss_p': loss_p,
                'encoding_indices': encoding_indices
            }
            return result_dict

    def encode(self):
        raise NotImplementedError

class Semi_VQ_VAE_Emb(VQ_VAE_Base):
    """
    计算zp, zt 并重构
    """
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, label_mat=None, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, **kwargs)
        self._init_model(bos, label_mat, **kwargs)

    def _init_model(self, bos, label_mat, **kwargs):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.emb_size,
                                                          self.gpu_tracker, encoder_name='distilling')
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing')
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, decompose_number=self.decompose_number,
                                        q_name='vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer,
                                                  **kwargs)

    def encode(self, inputs, pad_mask):
        """
        inputs:
            inputs: torch.tensor 3d, sequence of word embedding
            pad_mask torch.tensor
        """
        # 计算latent variable
        z_t, var_t = self.encoder_distill(inputs, pad_mask=pad_mask)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(inputs, pad_mask=pad_mask)  # [batch_size, emb_size]
        return z_t, var_t, z_p, var_p

    def recon_loss_function(self, result_dict, x, x_pad_mask):
        recon_x = result_dict['recon_result']
        if x_pad_mask is None:
            x = x[:, :, :]  # [batch_size, seq_len-1, emb_size]
            recon_x = recon_x[:, :, :]
            ce_loss = F.mse_loss(recon_x.float(), x.float())
        else:
            mask = x_pad_mask[:, :]
            mask = (mask.unsqueeze(2) == 1).to(recon_x.device)
            x = torch.masked_select(x[:, :, :], mask)  # [batch_size, seq_len-1, emb_size]
            recon_x = torch.masked_select(recon_x[:, :, :], mask)
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


class Semi_VQ_VAE_Idx(VQ_VAE_Base):
    def __init__(self, config, device, gpu_tracker, bos, emb_layer, label_mat=None, **kwargs):
        super().__init__(config, device, gpu_tracker, bos, emb_layer, **kwargs)
        self._init_model(bos, label_mat)

    def _init_model(self, bos, label_mat, **kwargs):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](self.config, self.device, self.emb_size,
                                                          self.gpu_tracker, encoder_name='distilling')
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](self.config, self.device, self.disper_size,
                                                             self.gpu_tracker, encoder_name='dispersing')
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](self.disper_num, self.disper_size, self.comit_coef, decompose_number=self.decompose_number,
                                        q_name='vq dispersing')
        # decoder layer
        self.decoder = Decoder[self.decoder_type](self.config, self.device, bos=bos, emb_layer=self.emb_layer, **kwargs)

    def encode(self, inputs, pad_mask):
        """
        inputs:
            inputs: torch.tensor 2d, sequence of id
            pad_mask torch.tensor
        """
        # 计算x
        x = self.emb_layer.emb_model.emb_from_idx(inputs).to(inputs.device)
        # 计算latent variable
        z_t, var_t = self.encoder_distill(x, pad_mask=pad_mask)  # [batch_size, emb_size]
        z_p, var_p = self.encoder_disperse(x, pad_mask=pad_mask)  # [batch_size, emb_size]
        return z_t, var_t, z_p, var_p

    def recon_loss_function(self, result_dict, x, x_pad_mask):
        recon_x = result_dict['recon_result']
        # loss_fn = CEFocalLoss().to(recon_x.device)
        loss_fn = nn.CrossEntropyLoss().to(recon_x.device)
        if x_pad_mask is None:
            x = x[:, :]  # [batch_size, seq_len-1, emb_size]
            recon_x = recon_x[:, :, :]
            final_ce_loss = loss_fn(recon_x, x)
        else:
            mask = x_pad_mask[:, :]
            x_mask = (mask == 1).to(recon_x.device)
            # [batch_size, seq_len-1, emb_size]
            masked_x = torch.masked_select(x[:, :].to(recon_x.device), x_mask)
            recon_prob_mask = (mask.unsqueeze(2) == 1).to(recon_x.device)
            masked_recon_x = torch.masked_select(recon_x[:, :, :], recon_prob_mask)
            masked_recon_x = masked_recon_x.reshape(-1, recon_x.shape[2])
            final_ce_loss = loss_fn(masked_recon_x, masked_x).mean()
            print('only use first word as loss')
            final_ce_loss = loss_fn(recon_x[:, 1, :], x[:, 1]).mean()
        # for i in range(recon_x.shape[0]):
        #     max_word_idx = torch.argmax(recon_x[i, 1, :], dim=0).item()
        #     print('origin x is', x[i, 1], 'prob is', recon_x[i, 1, :], recon_x[i, 1, x[i, 1].item()], 'max word is', max_word_idx, 'max word prob is', recon_x[i, 1, max_word_idx])
        # test loss on each words
        # for i in range(recon_x.shape[1]):
            # print('recon result', recon_x[:, i, :], x[:, i])
            # loss = loss_fn(recon_x[:, i, :].detach(), x[:, i].detach())
            # print('in %dth word, loss is' % i, loss)
            
        return final_ce_loss

    def loss_function(self, result_dict, x, x_pad_mask, recon_loss=True, vq_loss=True, **kwargs):
        '''
        x: origin input [batch_size, seq_len, emb_size]
        recon_x: [batch_size, seq_len, ntokens]
        '''
        if vq_loss is True:
            loss_p = result_dict['loss_p']
            final_vq_loss = self.vq_coef * loss_p
        if recon_loss is True:
            final_ce_loss = self.config.recon_coef * self.recon_loss_function(result_dict, x, x_pad_mask)
        if vq_loss is True and recon_loss is True:
            return final_vq_loss + final_ce_loss
        elif vq_loss is True and recon_loss is False:
            return final_vq_loss
        elif vq_loss is False and recon_loss is True:
            return final_ce_loss
        else:
            return None


Semi_VQ_VAE = {
    'VQ_VAE': Semi_VQ_VAE_Emb,
    'VQ_VAE_Idx': Semi_VQ_VAE_Idx,
}  # type: ignore
