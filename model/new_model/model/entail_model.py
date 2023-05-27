import numpy as np
from typing import List, Union, Tuple
from model.new_model.modules.classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.modules.encoder import Encoder
from model.new_model.modules.decoder import Decoder
from model.new_model.modules.vq import (
    ConcreteQuantizer,
    StraightForwardZ,
    DVQ,
    VQ,
    FixedVectorQuantizer,
    FixedVectorQuantizerClassifier
    )


class EntailmentModel(nn.Module):
    """
    Entailment Bert Model with BCE classification loss
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, emb_layer_p, adj_parent, adj_child, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.gpu_tracker = gpu_tracker
        # embeding layer
        self.emb_layer = emb_layer

        # size config
        self.k = config.class_num  # 一共有多少个类
        self.emb_size = config.emb_size  # input size
        self.distil_size = config.distil_size  # dislling latent size
        # encoder and decode type config
        self.encoder_type = config.encoder_type  # encoder type
        
        # idx setting
        self.pad_idx = self.emb_layer.emb_model.pad_idx
        self.eos_idx = self.emb_layer.emb_model.eos_idx

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        # layers
        print('fix model, before encoder model layer %s' % self.encoder_type)
        # self.gpu_tracker.track()
        # distilling encoder
        self.encoder = Encoder[self.encoder_type](config=self.config,
                                                  emb_layer=self.emb_layer,
                                                  output_size=self.distil_size,
                                                  encoder_name='distilling')
        self.classifier = nn.Linear(self.emb_size, 1)
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, input_idx, x_pad_mask=None, y=None, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        result_dict = self.encoder(input_idx)
        choosed_logits = result_dict['output_memory']

        # train binary
        prob = self.classifier(choosed_logits)
        # get loss while training
        if y is None:
            assert self.training is False, 'Train but not give y label'
            loss = None
        else:
            loss = self.loss_function(prob, y)
        
        result_dict = {
            'prob': prob,
            'loss': loss,
            'z_t': choosed_logits
        }
        return result_dict


class GenerateModel(nn.Module):
    """
    Generate model with depersing encoder and decoder
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, emb_layer_p, adj_parent, adj_child, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.label_matrix = label_matrix
        self.gpu_tracker = gpu_tracker
        self.emb_layer = emb_layer
        self.emb_layer_p = emb_layer_p
        self.adj_parent = adj_parent
        self.adj_child = adj_child

        # size configßß
        self.k = config.class_num  # 一共有多少个类
        self.emb_size = config.emb_size  # input size
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

        self._init_model(label_mat=self.label_matrix, **kwargs)

    def _init_model(self, label_mat, **kwargs):
        # dispersing encoder
        self.encoder_disperse = Encoder[self.encoder_p_type](config=self.config,
                                                             # device=self.device,
                                                             emb_layer=self.emb_layer_p,
                                                             output_size=self.disper_size,
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
        print('decoder')
        self.decoder = Decoder[self.decoder_type](config=self.config,
                                                  # device=self.device,
                                                  emb_layer=self.emb_layer,
                                                  **kwargs)
        print('end decoder')

    def forward(self, input_idx, pad_mask=None, only_encoder=False, y=None, label_matrix=None, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        result_dict = {}
        input = input_from_batch(input_idx, self.eos_idx, self.pad_idx)
        src = input['enc_in']
        # get dispersing output
        enc_outdict_p = self.encoder_disperse(src)
        result_dict['z_p'] = enc_outdict_p["pooled_memory"]
        
        # get dispersing vector quatnizer output
        quantizer_out_p = self.emb_p(inputs=enc_outdict_p["pooled_memory"])  # [N, 1, E]
        result_dict['quantizer_out_p'] = quantizer_out_p
        # print('z_t is', quantizer_out_t)
        
        if only_encoder is False:
            memory = quantizer_out_p['quantized_stack']
            # get decoder result
            decoder_output = self.decoder(inputs=input, memory=memory, encoding_indices=quantizer_out_p["encoding_indices"], **kwargs)

            # calculate loss
            if self.training:
                total_loss = self.calculate_loss(decoder_output=decoder_output,
                                                 quantizer_out_p=quantizer_out_p)
            
                result_dict['loss'] = total_loss
            result_dict['decoder_output'] = decoder_output
        
        return result_dict

    def calculate_loss(self, decoder_output, quantizer_out_p, **kwargs):
        total_loss = decoder_output['loss_reconstruct']
        logprobs = decoder_output['logprobs']
        
        if type(self.emb_p) == ConcreteQuantizer:
            actual_kl = quantizer_out_p["loss"]
            if actual_kl < (self.kl_fbp * logprobs.shape[0]):
                total_loss = total_loss
            else:
                total_loss = total_loss + self.kl_beta * actual_kl
            print('total_loss is', total_loss, 'vq loss is', self.kl_beta * actual_kl)
        elif type(self.emb_p) in [StraightForwardZ, FixedVectorQuantizer, FixedVectorQuantizerClassifier]:
            pass
        elif type(self.emb_p) == DVQ:
            total_loss = total_loss + quantizer_out_p["loss"]
        elif type(self.emb_t) in [FixedVectorQuantizer, FixedVectorQuantizerClassifier]:
            total_loss = total_loss + quantizer_out_p["loss"]
        else:
            raise TypeError
        
        return total_loss

    def sample_all(self, y_text, device, one_by_one=False):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        if self.p_vq_type in ['GS', 'DVQ'] and self.config.decompose_number > 1:
            raise NotImplementedError
        else:
            p_idx = torch.arange(0, self.disper_num, 1).to(device).unsqueeze(1)
            # print(p_idx)
            codebook_emb_p = self.emb_p.quantize_embedding(p_idx)  # [disper num, disperse_size]
            # result
            if one_by_one is True:
                result_dict = {}
                for i in range(codebook_emb_p.shape[0]):
                    one_result_dict = self.decoder(texts=y_text_list, memory=codebook_emb_p[i, :, :].unsqueeze(0),)  # [batch_size, emb_size]
                    if result_dict == {}:
                        result_dict['pred_idx'] = one_result_dict['pred_idx']
                    else:
                        result_dict['pred_idx'] = torch.cat([result_dict['pred_idx'], one_result_dict['pred_idx']], dim=0)
            else:
                y_text_list = [y_text for _ in range(codebook_emb_p.shape[0])]
                result_dict = self.decoder.generate_by_text(texts=y_text_list, memory=codebook_emb_p)  # [batch_size, emb_size]
        return result_dict, p_idx  # 截断梯度流
    
    def sample_vectors(self, index_list: List):
        """Sample disperse vectors for seen and unseen for discriminator training

        Args:
            index_list (_type_): _description_
        """
        p_idx = torch.tensor(index_list).unsqueeze(1)
        codebook_emb_p = self.emb_p.quantize_embedding(p_idx)
        return codebook_emb_p
        

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
