from turtle import forward
from numpy import choose
import torch
import torch.nn as nn
from model.gpt2model.model.gpt2 import MyGPTGenerate


class GPTClassifierModel(nn.Module):
    def __init__(self, model_config, device, label_matrix):
        super().__init__()
        self.emb_size = model_config.emb_size
        self.class_num = model_config.class_num
        self.label_matrix = label_matrix

        self.gpt2 = MyGPTGenerate(config=model_config, device=device)
        self.classifier_layer1 = nn.Linear(self.emb_size + label_matrix.shape[1], 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.loss_function = nn.CrossEntropyLoss()
        
    def forward(self, sentence, y):
        result_dict = self.gpt2(sentence)
        print('attention mask, ', result_dict['label_attn_mask'])
        choosed_idx = result_dict['label_attn_mask'].sum(dim=1)
        choosed_idx = choosed_idx - 1
        print(result_dict['logits'].shape, choosed_idx.shape, choosed_idx)
        # choosed_logits = result_dict['logits'][:, choosed_idx, :].contiguous()
        idx_list = [i for i in range(result_dict['logits'].shape[0])]
        choosed_logits = result_dict['logits'][torch.tensor(idx_list), choosed_idx, :].contiguous()
        print('check choosed_logits', choosed_logits.shape, choosed_logits[3], result_dict['logits'][3, 79, :])
        
        label_mat = self.label_matrix.unsqueeze(0).repeat(choosed_logits.shape[0], 1, 1).to(choosed_logits.device)
        cls_num = label_mat.shape[1]  # label count
        
        z_t_mat = choosed_logits.unsqueeze(1).repeat(1, cls_num, 1)  # [batch, label_count, distil_size]
        print('z_t_mat shape is', z_t_mat.shape, 'label_mat shape is', label_mat.shape)
        relation_pairs = torch.cat([z_t_mat, label_mat], 2)
        prob = self.classifier_layer1(relation_pairs)  # [batch * label_count, 256]
        prob = self.fc2(self.relu(prob))
        prob = prob.reshape(choosed_logits.shape[0], -1)
        
        loss = self.loss_function(prob, y)
        
        result_dict = {
            'prob': prob,
            'loss': loss
        }
        return result_dict
        