#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import torch.nn as nn

from model.attentionxml.deepxml.modules import LSTMEncoder, MLAttention, MLLinear, FastMLAttention


__all__ = ['AttentionRNN', 'FastAttentionRNN']


class Network(nn.Module):
    """

    """
    def __init__(self, model_config):
        super(Network, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    """

    """
    def __init__(self, model_config):
        super(AttentionRNN, self).__init__(model_config)
        self.model_config = model_config
        self.labels_num = model_config.class_num
        self.emb_size = model_config.emb_size
        self.hidden_size = model_config.hidden_size
        self.layers_num = model_config.layers_num
        self.dropout = model_config.dropout_rate
        self.linear_size = model_config.linear_size

        self.lstm = LSTMEncoder(self.emb_size, self.hidden_size, self.layers_num, self.dropout)
        self.attention = MLAttention(self.labels_num, self.hidden_size * 2)
        self.linear = MLLinear([self.hidden_size * 2] + self.linear_size, 1)

    def forward(self, emb_out, lengths, masks):
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks)      # N, labels_num, hidden_size * 2
        return self.linear(attn_out)


class FastAttentionRNN(Network):
    """

    """
    def __init__(self, model_config, emb_layer, label_mat, labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, parallel_attn, **kwargs):
        super(FastAttentionRNN, self).__init__(emb_size, **kwargs)
        self.model_config = model_config
        self.labels_num = model_config.class_num
        self.emb_size = model_config.emb_size
        self.hidden_size = model_config.hidden_size
        self.layers_num = model_config.layers_num
        self.dropout = model_config.dropout_rate
        self.linear_size = model_config.linear_size

        self.lstm = LSTMEncoder(self.emb_size, self.hidden_size, self.layers_num, self.dropout)
        self.attention = FastMLAttention(self.labels_num, self.hidden_size * 2, parallel_attn)
        self.linear = MLLinear([self.hidden_size * 2] + self.linear_size, 1)

    def forward(self, inputs, candidates, attn_weights: nn.Module, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks, candidates, attn_weights)     # N, sampled_size, hidden_size * 2
        return self.linear(attn_out)
