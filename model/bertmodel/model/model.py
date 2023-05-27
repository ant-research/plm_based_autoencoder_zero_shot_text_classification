# Bert 推理模型
from turtle import distance, forward
import torch
import torch.nn as nn
from model.bertmodel.model.bert import Bert_CLS


class BertClassifierModel(nn.Module):
    def __init__(self, model_config, emb_layer, label_matrix):
        super().__init__()
        self.emb_size = model_config.emb_size
        self.label_matrix = label_matrix

        self.bert = Bert_CLS(emb_layer=emb_layer)
        self.bert.train()
        
        self.classifier = nn.Linear(self.emb_size, 1)
        # self.classifier_layer1 = nn.Linear(self.emb_size + label_matrix.shape[1], 256)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(256, 1)
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, y=None):
        result_dict = self.bert(inputs)
        choosed_logits = result_dict['output_memory']
        print(choosed_logits.shape, self.label_matrix.shape)
        
        # train with relation network
        # choosed_logits = choosed_logits.unsqueeze(1)
        # label_mat = self.label_matrix.unsqueeze(0).repeat(choosed_logits.shape[0], 1, 1).to(choosed_logits.device)
        # cls_num = label_mat.shape[1]  # label count
        # z_t_mat = choosed_logits.repeat(1, cls_num, 1)  # [batch, label_count, distil_size]
        # relation_pairs = torch.cat([z_t_mat, label_mat], 2)
        # prob = self.classifier_layer1(relation_pairs)  # [batch * label_count, 256]
        # prob = self.fc2(self.relu(prob))
        # prob = prob.reshape(choosed_logits.shape[0], -1)
        
        # train with distance
        # label_mat = self.label_matrix.to(choosed_logits.device)
        # distances = (
        #     torch.sum(choosed_logits ** 2, dim=1, keepdim=True)
        #     + torch.sum(label_mat ** 2, dim=1)
        #     - 2 * torch.matmul(choosed_logits, label_mat.t())
        # )
        # prob = -distances
        
        # train binary
        prob = self.classifier(choosed_logits)
        if y is None:
            loss = 0
        else:
            loss = self.loss_function(prob, y)
        
        result_dict = {
            'prob': prob,
            'loss': loss
        }
        return result_dict
        