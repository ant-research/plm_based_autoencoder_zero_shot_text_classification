# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()

    def forward(self, z, z_da, **kwargs):
        '''
        z: [batch, emb_size]
        z_da: [batch, emb_size]
        '''
        z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
        z_da_norm = torch.norm(z_da, p=2, dim=1, keepdim=True)
        z = z/z_norm
        z_da = z_da/z_da_norm
        cos_self = torch.matmul(z, z.t())
        cos_other = torch.matmul(z, z_da.t())
        cos_other_diag = torch.diag_embed(torch.diag(cos_other)) # [batch_size, batch_size]
        cos_self_diag = torch.diag_embed(torch.diag(cos_self)) # [batch_size, batch_size]
        cos_all = cos_self - cos_self_diag + cos_other_diag
        cos_all = torch.exp(cos_all)  # [batch_size, batch_size]
        loss = -torch.log(cos_all/torch.sum(cos_all, dim=1))
        loss = torch.mean(torch.diag(loss))
        return loss


class ContrastiveLoss_Neg(nn.Module):
    """
    Only negative Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, **kwargs):
        super(ContrastiveLoss_Neg, self).__init__()

    def forward(self, z, label_mat, **kwargs):
        '''
        计算z和label mat之间的相似度并求倒数的负数
        Input:
            z_p: [batch, emb_size]
            label_mat: [label_num, emb_size]
        '''
        # calculate cos similarity
        z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
        z = z/z_norm
        label_mat = label_mat/(torch.norm(label_mat, p=2, dim=1, keepdim=True) + 1e-4)
        # print('z result', z)
        # print('label_mat result', label_mat)
        cos_self = torch.exp(torch.matmul(z, label_mat.t().to(z.device)))
        # print('cosine result', cos_self)
        loss = -torch.log(1/(torch.sum(cos_self, dim=1)+1))
        loss = torch.mean(loss)
        return loss


class GraphCELoss(nn.Module):
    def __init__(self, adj_parent, **kwargs):
        super().__init__()
        adj_parent = adj_parent - torch.eye(adj_parent.shape[0]) 

        
        def get_parent_node(adj_parent, i):
            now_vec = adj_parent[i, :]
            if now_vec.sum() == 0: # find the root node
                return now_vec
            else:
                parent_idx = torch.argmax(now_vec)
                parent_vec = get_parent_node(adj_parent, parent_idx)
                return now_vec + parent_vec

        masked_list = []
        for i in range(adj_parent.shape[0]):
            masked_list.append(get_parent_node(adj_parent, i).unsqueeze(0))
            
        masked_mat = torch.cat(masked_list, dim=0)
        self.register_buffer('masked_mat', masked_mat)

        self.log_softmax = nn.LogSoftmax()
        self.nll_loss = nn.NLLLoss()

    def forward(self, logits, target, **kwargs):
        '''
        计算z和label mat之间的相似度并求倒数的负数
        Input:
            z_p: [batch, emb_size]
            label_mat: [label_num, emb_size]
        '''
        mask = self.masked_mat[target, :]  # type: ignore [batch, N]
        logits = logits.masked_fill(mask == 1, float('-inf'))
        log_probabilities = self.log_softmax(logits)
        print('log_probabilities is', log_probabilities)
        # NLLLoss(x, class) = -weights[class] * x[class]
        return self.nll_loss(log_probabilities, target)


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CEFocalLoss(nn.Module):
    '''
    import two probability
    probs: [batch size, class number]
    target: [batch size, class number]
    '''
    def __init__(self, weight=None, reduction='mean', gamma=2):
        super(CEFocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class CELoss_NoSoftmax(nn.Module):
    '''
    import two probability
    probs: [batch size, class number]
    target: [batch size, class number]
    '''
    def __init__(self):
        super().__init__()
        self.nll_func = nn.NLLLoss()

    def forward(self, inputs, target):
        print('loss input is', inputs)
        print(target)
        log_input = torch.log(inputs)
        loss = self.nll_func(log_input, target)
        return loss
