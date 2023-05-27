from base64 import decode
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.modules.encoder import Encoder
from model.new_model.modules.decoder import Decoder
from model.new_model.utils.loss import CEFocalLoss

import numpy as np

from model.new_model.modules.vq import (
    ConcreteQuantizer,
    StraightForwardZ,
    DVQ,
    VQ,
    FixedVectorQuantizer,
    FixedVectorQuantizerClassifier
    )
# from model.new_model.utils.vq import VQ


class VQVAEBase(nn.Module):
    def __init__(self, config, device, gpu_tracker, emb_layer, emb_layer_p, label_mat, **kwargs):
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

        self.disper_num = config.disper_num
        # vq type config
        self.t_vq_type = config.distil_vq_type  # Soft or VQ or Fix
        self.p_vq_type = config.disperse_vq_type  # Soft or VQ or Fix
        # encoder and decode type config
        self.encoder_type = config.encoder_type  # encoder type
        try:
            self.encoder_p_type = config.encoder_p_type
            if self.encoder_p_type in ['GS']:
                self.disper_size = self.k
            else:
                self.disper_size = config.disper_size  # dispersing latent size
        except Exception as e:
            print(e, 'not encoder p type')
            self.encoder_p_type = config.encoder_type
        self.decoder_type = config.decoder_type  # decoder type
        # setting loss coefficient
        self.vq_coef = config.vq_coef
        self.comit_coef = config.comit_coef
        # concrete loss coefficient
        self.kl_fbp = config.concrete_kl_fbp_threshold
        self.kl_beta = config.concrete_kl_beta
        
        # embeding layer
        self.emb_layer = emb_layer
        self.emb_layer_p = emb_layer_p
        # idx setting
        self.pad_idx = self.emb_layer.emb_model.pad_idx
        self.eos_idx = self.emb_layer.emb_model.eos_idx
        


        self._init_model(label_mat, **kwargs)

    def _init_model(self, label_mat, **kwargs):
        # layers
        print('before encoder model layer')
        self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](config=self.config,
                                                          device=self.device,
                                                          emb_layer=self.emb_layer,
                                                          output_size=self.distil_size,
                                                          gpu_tracker=self.gpu_tracker,
                                                          encoder_name='distilling',
                                                          vq_type=self.t_vq_type)
        print('after encoder model layer')
        self.gpu_tracker.track()
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](config=self.config,
                                                             device=self.device,
                                                             emb_layer=self.emb_layer_p,
                                                             output_size=self.disper_size,
                                                             gpu_tracker=self.gpu_tracker,
                                                             encoder_name='dispersing',
                                                             vq_type=self.p_vq_type)
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](num_embeddings=self.disper_num,
                                        embedding_dim=self.disper_size,
                                        config=self.config,
                                        q_name='vq dispersing',
                                        label_mat=label_mat)
        
        # decoder layer
        self.decoder = Decoder[self.decoder_type](config=self.config,
                                                  device=self.device,
                                                  emb_layer=self.emb_layer,
                                                  encoder=self.encoder_distill,
                                                  **kwargs)

    def forward(self, inputs_batch, pad_mask=None, only_encoder=False, y=None, label_matrix=None, **kwargs):
        print('before model used memory', torch.cuda.mem_get_info())
        result_dict = {}
        input = input_from_batch(inputs_batch, self.eos_idx, self.pad_idx)
        src = input['enc_in']

        # encoder
        enc_outdict_t = self.encoder_distill(src)
        result_dict['z_t'] = enc_outdict_t["pooled_memory"]
        print('after distilling encoder memory', torch.cuda.mem_get_info())
        result_dict['enc_outdict_t'] = enc_outdict_t

        enc_outdict_p = self.encoder_disperse(src)
        print('after dispersing encoder memory', torch.cuda.mem_get_info())

        # get quantizer
        quantizer_out_t = enc_outdict_t["pooled_memory"]
        quantizer_out_p = self.emb_p(enc_outdict_p["pooled_memory"])
        result_dict['z_p'] = quantizer_out_p['quantized_stack']
        result_dict['quantizer_out_t'] = quantizer_out_t
        result_dict['quantizer_out_p'] = quantizer_out_p

        if only_encoder is False:
            # print('z_t is', quantizer_out_t)
            # print('z_p is', quantizer_out_p['quantized_stack'])
            
            result_dict['quantizer_out_t'] = quantizer_out_t
            result_dict['quantizer_out_p'] = quantizer_out_p
            # print('after quantized memory', torch.cuda.mem_get_info())
            
            if y is None:
                quantizer_out_t = enc_outdict_t["pooled_memory"]
            else:
                print('use true y label', y, label_matrix)
                quantizer_out_t = label_matrix[y, :]
            
            memory = torch.cat([quantizer_out_t.unsqueeze(1), quantizer_out_p['quantized_stack']], dim=1)
            if self.config.decoder_input_size != self.config.encoder_output_size:
                memory = self.proj_layer(memory)
            # get decoder result
            decoder_output = self.decoder(inputs=input, memory=memory, encoding_indices=quantizer_out_p["encoding_indices"], **kwargs)
            # print('after decoder memory', torch.cuda.mem_get_info())

            # calculate loss
            if self.training:
                total_loss = self.calculate_loss(decoder_output=decoder_output,
                                                 quantizer_out_p=quantizer_out_p)
            
                result_dict['loss'] = total_loss
            result_dict['decoder_output'] = decoder_output
        
        return result_dict

    def calculate_loss(self, decoder_output, quantizer_out_p):
        total_loss = decoder_output['loss_reconstruct']
        logprobs = decoder_output['logprobs']
        
        if type(self.emb_p) == ConcreteQuantizer:
            actual_kl = quantizer_out_p["loss"]
            
            if actual_kl < (self.kl_fbp * logprobs.shape[0]):
                total_loss = total_loss
            else:
                total_loss = total_loss + self.kl_beta * actual_kl
        elif type(self.emb_p) == StraightForwardZ:
            pass
        elif type(self.emb_p) == DVQ:
            total_loss = total_loss + quantizer_out_p["loss"]
        else:
            raise TypeError
        
        return total_loss


class FixVQVAE(VQVAEBase):
    # Fix住label vae
    def __init__(self, config, device, gpu_tracker, emb_layer, emb_layer_p, label_mat, **kwargs):
        super().__init__(config, device, gpu_tracker, emb_layer, emb_layer_p, label_mat, **kwargs)

    def _init_model(self, label_mat, **kwargs):
        # layers
        print('fix model, before encoder model layer %s' % self.encoder_type)
        # self.gpu_tracker.track()
        # distilling encoder
        self.encoder_distill = Encoder[self.encoder_type](config=self.config,
                                                          # device=self.device,
                                                          emb_layer=self.emb_layer,
                                                          output_size=self.distil_size,
                                                          encoder_name='distilling',
                                                          vq_type=self.t_vq_type)
        print('after encoder model layer')
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](config=self.config,
                                                             # device=self.device,
                                                             emb_layer=self.emb_layer_p,
                                                             output_size=self.disper_size,
                                                             encoder_name='dispersing',
                                                             vq_type=self.p_vq_type)
        # distilling vq embedding
        print('distilling vq type is', self.t_vq_type)
        self.emb_t = VQ[self.t_vq_type](num_embeddings=self.disper_num,
                                        embedding_dim=self.distil_size,
                                        config=self.config,
                                        q_name='vq distilling',
                                        label_mat=label_mat)
        # dispersing vq embedding
        print('dispersing vq type is', self.p_vq_type)
        self.emb_p = VQ[self.p_vq_type](num_embeddings=self.disper_num,
                                        embedding_dim=self.disper_size,
                                        config=self.config,
                                        q_name='vq dispersing',
                                        label_mat=label_mat)
        # decoder layer
        print('decoder')
        self.decoder = Decoder[self.decoder_type](config=self.config,
                                                  # device=self.device,
                                                  emb_layer=self.emb_layer,
                                                  encoder=self.encoder_distill,
                                                  **kwargs)
        print('end decoder')

    def forward(self, inputs_batch, pad_mask=None, only_encoder=False, y=None, label_matrix=None, **kwargs):
        print('before model used memory', torch.cuda.mem_get_info())
        result_dict = {}
        input = input_from_batch(inputs_batch, self.eos_idx, self.pad_idx)
        src = input['enc_in']

        # encoder
        enc_outdict_t = self.encoder_distill(src)
        print('after distilling encoder memory', torch.cuda.mem_get_info())
        result_dict['enc_outdict_t'] = enc_outdict_t

        enc_outdict_p = self.encoder_disperse(src)
        print('after dispersing encoder memory', torch.cuda.mem_get_info())

        # get quantizer
        quantizer_out_t = self.emb_t(inputs=enc_outdict_t["pooled_memory"], label_matrix=label_matrix, y=y)  # [N, 1, E] 
        quantizer_out_p = self.emb_p(inputs=enc_outdict_p["pooled_memory"])  # [N, 1, E]
        # print('z_t is', quantizer_out_t)
        # print('z_p is', quantizer_out_p['quantized_stack'])
        result_dict['z_t'] = enc_outdict_t["pooled_memory"]
        result_dict['z_p'] = enc_outdict_p["pooled_memory"]

        if only_encoder is False:
            result_dict['quantizer_out_t'] = quantizer_out_t
            result_dict['quantizer_out_p'] = quantizer_out_p
            # print('after quantized memory', torch.cuda.mem_get_info())
            if y is None:
                z_t = quantizer_out_t['quantized_stack']
            else:
                # print('use true y label', y, label_matrix)
                z_t = label_matrix[y, :].unsqueeze(1)

            memory = torch.cat([z_t, quantizer_out_p['quantized_stack']], dim=1)
            
            # get decoder result
            decoder_output = self.decoder(inputs=input, memory=memory, encoding_indices=quantizer_out_p["encoding_indices"], **kwargs)
            # print('after decoder memory', torch.cuda.mem_get_info())

            # calculate loss
            if self.training:
                total_loss = self.calculate_loss(decoder_output=decoder_output,
                                                 quantizer_out_p=quantizer_out_p,
                                                 quantizer_out_t=quantizer_out_t)
            
                result_dict['loss'] = total_loss
            result_dict['decoder_output'] = decoder_output
        
        return result_dict

    def calculate_loss(self, decoder_output, quantizer_out_p, quantizer_out_t, **kwargs):
        total_loss = decoder_output['loss_reconstruct']
        logprobs = decoder_output['logprobs']
        
        if type(self.emb_p) == ConcreteQuantizer:
            actual_kl = quantizer_out_p["loss"]
            
            if actual_kl < (self.kl_fbp * logprobs.shape[0]):
                total_loss = total_loss
            else:
                total_loss = total_loss + self.kl_beta * actual_kl
        elif type(self.emb_p) in [StraightForwardZ, FixedVectorQuantizer, FixedVectorQuantizerClassifier]:
            pass
        elif type(self.emb_p) == DVQ:
            total_loss = total_loss + quantizer_out_p["loss"]
        elif type(self.emb_t) in [FixedVectorQuantizer, FixedVectorQuantizerClassifier]:
            total_loss = total_loss + quantizer_out_p["loss"]
        else:
            raise TypeError
        
        if type(self.emb_t) == ConcreteQuantizer:
            actual_kl = quantizer_out_t["loss"]
            
            if actual_kl < (self.kl_fbp * logprobs.shape[0]):
                total_loss = total_loss
            else:
                total_loss = total_loss + self.kl_beta * actual_kl
        elif type(self.emb_t) in [StraightForwardZ]:
            pass
        elif type(self.emb_t) in [DVQ]:
            total_loss = total_loss + quantizer_out_t["loss"]
        elif type(self.emb_t) in [FixedVectorQuantizer, FixedVectorQuantizerClassifier]:
            total_loss = total_loss + quantizer_out_t["loss"]
        else:
            raise TypeError
        
        return total_loss


def input_from_batch(inputs, eos_idx, pad_idx):
    sent = inputs  # shape (batch_size, seq_len)
    batch_size, seq_len = sent.size()

    # no <SOS> and <EOS>
    enc_in = sent[:, :].clone()
    enc_in[enc_in == eos_idx] = pad_idx

    # no <SOS>
    dec_out_gold = sent[:, 1:].contiguous()

    # no <EOS>
    dec_in = sent[:, :-1].clone()
    dec_in[dec_in == eos_idx] = pad_idx

    out = {
        "batch_size": batch_size,
        "dec_in": dec_in,
        "dec_out_gold": dec_out_gold,
        "enc_in": enc_in,
        "sent": sent,
    }
    return out

