import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint


class Ernie(nn.Module):
    def __init__(self, config, device, output_size, gpu_tracker, emb_layer, encoder_name='ernie encoder'):
        super().__init__()
        self.gpu_tracker = gpu_tracker
        self.encoder_name = encoder_name
        self.output_size = output_size
        self.k = config.class_num
        self.emb_size = config.emb_size
        self.device = device
        self.ernie_layer = emb_layer
        self.ernie_model = self.ernie_layer.emb_model.model
        self.output_layer = nn.Linear(self.emb_size, self.output_size)
        self.var_layer = nn.Linear(self.emb_size, self.k)

    def forward(self, inputs, pad_mask=None, targets=None, detach=False):
        outputs = self.ernie_model(inputs)
        # get the CLS value in the last layer
        hidden_states = outputs['hidden_states']
        embs = hidden_states[-1][:, 0, :]
        if detach:
            embs = embs.detach()
        else:
            pass
        output = self.output_layer(embs)
        var = self.var_layer(embs)
        return output, var


class Ernie_Disperse(nn.Module):
    def __init__(self, config, device, output_size, gpu_tracker, emb_layer, encoder_name='ernie encoder'):
        super().__init__()
        self.gpu_tracker = gpu_tracker
        self.encoder_name = encoder_name
        self.output_size = output_size
        self.k = config.class_num
        self.emb_size = config.emb_size
        self.max_len = config.seq_len
        self.device = device
        self.ernie_layer = emb_layer
        self.ernie_model = self.ernie_layer.emb_model.model
        self.conv_layer = nn.Linear(self.max_len, 1)
        self.output_layer = nn.Linear(self.emb_size, self.output_size)
        self.var_layer = nn.Linear(self.emb_size, self.k)

    def forward(self, inputs, pad_mask=None, targets=None, detach=False):
        outputs = self.ernie_model(inputs)
        # get the CLS value in the last layer
        hidden_states = outputs['hidden_states']
        embs = hidden_states[-1][:, :, :]
        print('embs shape', embs.shape)
        if detach:
            embs = embs.detach()
        else:
            pass
        embs = self.conv_layer(embs.permute(0, 2, 1)).squeeze(2)
        print('embs shape', embs.shape)
        output = self.output_layer(embs)
        var = self.var_layer(embs)
        return output, var


Encoder = {
    'Ernie': Ernie,
    'Ernie_P': Ernie_Disperse
}
