import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from model.new_model.modules.simple_modules import Pooler


class Bert_CLS(nn.Module):
    def __init__(self, config, emb_layer, vq_type, encoder_name='bert encoder', **kwargs):
        print('start build encoder')
        super().__init__()
        print('start init super cls')
        self.encoder_name = encoder_name
        # 输入参数        
        self.ernie_layer = emb_layer
        self.ernie_model = self.ernie_layer.emb_model.model
        print('set require grad false')
        for p in self.parameters():
            p.requires_grad = False

        self.pad_idx = emb_layer.emb_model.pad_idx

        # bert 后面加两层
        print('add new bert')
        self.hid_emb = config.emb_size
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.forward_expansion = config.forward_expansion

        # output
        self.vq_type = vq_type
        self.output_dim = config.encoder_output_size
        
        encoder_layers = TransformerEncoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                 dim_feedforward=self.forward_expansion*self.hid_emb,
                                                 dropout=self.dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, 2)

        self.output_layer = nn.Linear(self.hid_emb, self.output_dim)
        print('finish bert build')

    def forward(self, inputs, pad_mask=None, targets=None, detach=False):
        #  positions with the value of True will be ignored while the position with the value of False will be unchanged
        src_pad_mask = inputs == self.pad_idx
        src_nopad_mask = inputs != self.pad_idx   # 0 for token that are masked
        
        outputs = self.ernie_model(inputs, attention_mask=src_nopad_mask)
        # get the CLS value in the last layer
        hidden_states = outputs['hidden_states']
        embs = hidden_states[-1].transpose(0, 1).detach() # last layer [batch size, seq len, hid dim]
        
        memory = self.encoder(embs, src_key_padding_mask=src_pad_mask).transpose(0, 1)
        # print(self.encoder_name, 'inputs is', inputs, src_nopad_mask, src_pad_mask)
        # print(self.encoder_name, 'bert output memory shape', memory.shape)
        memory = memory[:, 0, :]
        pooled_memory = self.output_layer(memory)
        # print(self.encoder_name, 'pooled memory is', pooled_memory)

        result_dict = {
            "pooled_memory": pooled_memory,
            "nopad_mask": src_nopad_mask
        }
        return result_dict


class Bert_Pooler(nn.Module):
    def __init__(self, config, emb_layer, vq_type, encoder_name='bert encoder', **kwargs):
        print('start build encoder')
        super().__init__()
        print('end init super pooler')
        self.encoder_name = encoder_name
        # 输入参数
        self.ernie_layer = emb_layer
        self.ernie_model = self.ernie_layer.emb_model.model
        for p in self.parameters():
            p.requires_grad = False
        self.pad_idx = emb_layer.emb_model.pad_idx
        
        # bert 后面加两层
        self.hid_emb = config.emb_size
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.forward_expansion = config.forward_expansion
        
        # output
        self.vq_type = vq_type
        self.output_dim = config.encoder_output_size
        
        encoder_layers = TransformerEncoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                 dim_feedforward=self.forward_expansion*self.hid_emb,
                                                 dropout=self.dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, 2)
        
        # sentence
        if self.vq_type in ["Striaght", "DVQ"]:
            d_proj = config.decompose_number * self.output_dim  # output dimension * decompose number
        elif self.vq_type in ['GS']:
            d_proj = config.decompose_number * config.disper_num
        else:
            raise KeyError
        
        self.pooler = Pooler(
            project=True,
            pool_type="mean",
            d_inp=self.hid_emb,
            d_proj=d_proj,
        )

    def forward(self, inputs, pad_mask=None, targets=None, detach=False):
        #  positions with the value of True will be ignored while the position with the value of False will be unchanged
        src_pad_mask = inputs == self.pad_idx
        src_nopad_mask = inputs != self.pad_idx   # 0 for token that are masked
        outputs = self.ernie_model(inputs, attention_mask=src_nopad_mask)
        # get the CLS value in the last layer
        hidden_states = outputs['hidden_states']
        embs = hidden_states[-1].transpose(0, 1).detach() # last layer [batch size, seq len, hid dim]

        memory = self.encoder(embs, src_key_padding_mask=src_pad_mask).transpose(0, 1)
        pooled_memory = self.pooler(memory, src_nopad_mask)  # [B, E*decompose_number]
        # print('pooled memory is', pooled_memory)

        result_dict = {
            "pooled_memory": pooled_memory,
            "nopad_mask": src_nopad_mask
        }
        return result_dict


class Bert_Entail_CLS(nn.Module):
    def __init__(self, config, emb_layer, encoder_name='bert encoder', **kwargs):
        super().__init__()
        self.encoder_name = encoder_name
        # 输入参数        
        self.ernie_layer = emb_layer
        self.ernie_model = self.ernie_layer.emb_model.model
        self.pad_idx = emb_layer.emb_model.pad_idx
        
        for name, param in self.named_parameters():
            print(name)
            if '.cls' in name:
                param.requires_grad = False

    def forward(self, inputs, pad_mask=None, targets=None, detach=False):
        #  positions with the value of True will be ignored while the position with the value of False will be unchanged
        src_pad_mask = inputs == self.pad_idx
        src_nopad_mask = inputs != self.pad_idx   # 0 for token that are masked
        
        outputs = self.ernie_model(inputs, attention_mask=src_nopad_mask)
        # get the CLS value in the last layer
        hidden_states = outputs['hidden_states']
        memory = hidden_states[-1][:, 0, :] # last layer [batch size, seq len, hid dim]
        result_dict = {
            'output_memory': memory
        }
        return result_dict