import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Bert_CLS(nn.Module):
    def __init__(self, emb_layer, **kwargs):
        super().__init__()
        # 输入参数
        self.ernie_model = AutoModel.from_pretrained('./pretrained_model/bert_t', output_hidden_states=True)
        self.pad_idx = emb_layer.emb_model.pad_idx
        for name, param in self.named_parameters():
            print(name)
            if 'pooler' in name:
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