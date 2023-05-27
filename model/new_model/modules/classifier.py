from turtle import distance
import torch
import torch.nn as nn
from typing import Union
from model.new_model.utils.loss import BCEFocalLoss, GraphCELoss


class ClassifierBase(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        if config.task_type == 'Multi_Label':
            self.loss_func = BCEFocalLoss(gamma=config.gamma)
            # self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def loss_function(self, relation, y):
        '''
        '''
        # print('y is', y)
        d_loss = self.loss_func(relation, y)
        return d_loss


class RelationNet(ClassifierBase):
    def __init__(self, config, label_mat, **kwargs):
        super().__init__(config, **kwargs)
        # build label matrix
        if label_mat.shape[1] == 1:
            self.label_mat = label_mat.squeeze(1)
        else:
            self.label_mat = label_mat

        self.emb_size = config.emb_size
        self.distil_size = config.distil_size
        # self.label_fc = nn.Linear(self.emb_size, self.emb_size)
        self.fc1 = nn.Linear(self.emb_size + self.distil_size, 600)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(600, 1)

        if config.task_type == 'Multi_Label':
            self.loss_func = BCEFocalLoss(gamma=config.gamma)
            # self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, **kwargs):
        '''
        z: [batch, distil_size]
        label_mat: [label_count, emb_size]
        '''
        # label_mat = self.label_fc(self.label_mat)
        label_mat = self.label_mat.unsqueeze(0).repeat(z_t.shape[0], 1, 1).to(z_t.device)
        cls_num = label_mat.shape[1]  # label count
        z_t_mat = z_t.unsqueeze(1).repeat(1, cls_num, 1)  # [batch, label_count, distil_size]
        relation_pairs = torch.cat([z_t_mat, label_mat], 2)
        relation_2 = self.relu(self.fc1(relation_pairs))  # [batch * label_count, 256]

        relation = self.fc2(relation_2).squeeze()  # [batch * label_count, 1]

        relation = relation.reshape(z_t.shape[0], -1)

        return relation

    def loss_function(self, relation, y):
        '''
        '''
        # print('y is', y)
        d_loss = self.loss_func(relation, y)
        return d_loss


class LinearClassifier(ClassifierBase):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.distil_size = config.distil_size
        self.class_num = config.class_num
        self.fc1 = nn.Linear(self.distil_size, 600)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(600, config.class_num)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, label_mat=None):
        '''
        z: [batch, distil_size]
        label_mat: [label_count, emb_size]
        '''
        # result_m = z_t/torch.norm(z_t, p=2, dim=1, keepdim=True)
        # label_cos = torch.matmul(result_m, result_m.t())
        # print('z_t before relu cos similarity', label_cos)

        z_t = self.relu(self.fc1(z_t))

        # result_m = z_t/torch.norm(z_t, p=2, dim=1, keepdim=True)
        # label_cos = torch.matmul(result_m, result_m.t())
        # print('z_t after relu cos similarity', label_cos)

        relation = self.fc2(z_t)

        # relation = self.sigmoid(relation)
        # print('relation max is', torch.max(relation, dim=1))
        # print('relation min is', torch.min(relation, dim=1))
        return relation


class LinearClassifierOld(ClassifierBase):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.distil_size = config.distil_size
        self.class_num = config.class_num
        self.fcs = nn.ModuleList([nn.Linear(self.distil_size, 1) for code in range(self.class_num)])
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, label_mat=None):
        '''
        z: [batch, distil_size]
        label_mat: [label_count, emb_size]
        '''
        relation = torch.zeros((z_t.size(0), self.class_num)).to(z_t.device)
        for code, fc in enumerate(self.fcs):
            relation[:, code:code+1] = fc(z_t)
        # relation = self.sigmoid(relation)
        print('linear relation max is', torch.max(relation, dim=1))
        print('linear relation min is', torch.min(relation, dim=1))
        return relation


class EuclideanClassifier(ClassifierBase):
    """
    Euclidean distance classifier
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)

    def forward(self, inputs: torch.Tensor, label_mat: torch.Tensor, y: Union[torch.Tensor, None] = None) -> dict:
        """
        calculate euclidean distance between input z and label matrix
        inputs:
            inputs: (2d tensor) [B, E]
            label_mat: (2d tensor) [label number, E]
        """
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True) +
            torch.sum(label_mat ** 2, dim=1) -
            2. * torch.matmul(inputs, label_mat.t())
        )  # [batch_size, K]
        result = {
            'prob': -distances
        }
        # print(self.training, y, y is None)
        if (self.training) and ((y is None) == False):
            loss = self.loss_func(-distances, y)
            result['loss'] = loss
        return result
    
class EuclideanClassifierWithGraph(ClassifierBase):
    """
    Euclidean distance classifier
    """
    def __init__(self, config, adj_parent, **kwargs):
        super().__init__(config)
        if config.task_type == 'Multi_Label':
            self.loss_func = BCEFocalLoss(gamma=config.gamma)
            # self.loss_func = nn.BCELoss()
        else:
            self.loss_func = GraphCELoss(adj_parent=adj_parent)

    def forward(self, inputs: torch.Tensor, label_mat: torch.Tensor, y: Union[torch.Tensor, None] = None) -> dict:
        """
        calculate euclidean distance between input z and label matrix
        inputs:
            inputs: (2d tensor) [B, E]
            label_mat: (2d tensor) [label number, E]
        """
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True) +
            torch.sum(label_mat ** 2, dim=1) -
            2. * torch.matmul(inputs, label_mat.t())
        )  # [batch_size, K]
        result = {
            'prob': -distances
        }
        # print(self.training, y, y is None)
        if (self.training) and ((y is None) == False):
            loss = self.loss_func(-distances, y)
            result['loss'] = loss
        return result


Classifier = {
    'RelationNet': RelationNet,
    'Linear': LinearClassifier,
    'Euclidean': EuclideanClassifier,
    'EuclideanGraph': EuclideanClassifierWithGraph
}
