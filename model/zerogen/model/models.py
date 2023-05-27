import numpy as np
from typing import List, Union, Tuple
from model.new_model.modules.classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.new_model.modules.encoder import Bert_Entail_CLS
from model.zerogen.model.gpt2 import NewGPT


class EntailmentModel(nn.Module):
    """
    Entailment Bert Model with BCE classification loss
    """
    def __init__(self, config, device, emb_layer, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
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
        self.encoder = Bert_Entail_CLS(config=self.config,
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
    def __init__(self, config, device, emb_layer, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.emb_layer = emb_layer
        # size configßß
        self.k = config.class_num  # 一共有多少个类
        self.emb_size = config.emb_size  # input size

        self.decoder_type = config.decoder_type  # decoder type
        # embeding layer
        self.emb_layer = emb_layer
        # idx setting
        self.pad_idx = self.emb_layer.emb_model.pad_idx
        self.eos_idx = self.emb_layer.emb_model.eos_idx

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        # decoder layer
        print('decoder')
        self.decoder = NewGPT(config=self.config,
                              device=self.device,
                              emb_layer=self.emb_layer,
                              **kwargs)
        print('end decoder')

    def sample_all(self, y_text, generate_number):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        y_text_list = [y_text for _ in range(generate_number)]
        result_dict = self.decoder.generate_by_label(label=y_text_list, label_text=[None for i in y_text_list])  # [batch_size, emb_size]
        return result_dict

        

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
