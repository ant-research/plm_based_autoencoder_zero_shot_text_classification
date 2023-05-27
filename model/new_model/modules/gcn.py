import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, adj_parent, adj_child, label_matrix, hid_q=256):
        """
        RGCN
        Args:
            adj_parent (2d tensor): adjacency matrix of parent
            adj_child (2d tensor): adjacency matrix of child
            label_matrix (2d tensor): label embedding matrix
            hid_q (int): output dimension
        """
        super(GCN, self).__init__()
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        input_dim = label_matrix.shape[1]

        self.w_self = nn.parameter.Parameter(torch.Tensor(input_dim, hid_q))
        self.w_parent = nn.parameter.Parameter(torch.Tensor(input_dim, hid_q))
        self.w_child = nn.parameter.Parameter(torch.Tensor(input_dim, hid_q))
        self.bias = nn.parameter.Parameter(torch.Tensor(hid_q))
        nn.init.kaiming_normal_(self.w_self)
        nn.init.kaiming_normal_(self.w_parent)
        nn.init.kaiming_normal_(self.w_child)
        nn.init.constant_(self.bias, 0)

    def forward(self, label_matrix, train=True):
        """
        Args:
            label_matrix (2d tensor): label embedding matrix
        Returns:
            new_label_matrix (2d tensor): new label embedding matrix
        """
        label_matrix = label_matrix.to(self.adj_parent.device)
        parent_emb = torch.matmul(self.adj_parent, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        child_emb = torch.matmul(self.adj_child, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        res = torch.matmul(label_matrix, self.w_self) + torch.matmul(parent_emb, self.w_parent) + \
            torch.matmul(child_emb, self.w_child) + self.bias
        res = F.relu(res)
        return res


class GCNParent(nn.Module):
    def __init__(self, adj_parent, adj_child, label_matrix, hid_q=256):
        """
        RGCN
        Args:
            adj_parent (2d tensor): adjacency matrix of parent
            adj_child (2d tensor): adjacency matrix of child
            input_dim (int): intput dimension
            hid_q (int): output dimension
        """
        super(GCNParent, self).__init__()
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        input_dim = label_matrix.shape[1]

        self.w_self = nn.parameter.Parameter(torch.Tensor(input_dim, hid_q))
        self.w_parent = nn.parameter.Parameter(torch.Tensor(input_dim, hid_q))

        self.bias = nn.parameter.Parameter(torch.Tensor(hid_q))
        nn.init.kaiming_normal_(self.w_self)
        nn.init.kaiming_normal_(self.w_parent)

        nn.init.constant_(self.bias, 0)

    def forward(self, label_matrix, train=True):
        """
        Args:
            label_matrix (2d tensor): label embedding matrix
        Returns:
            new_label_matrix (2d tensor): new label embedding matrix
        """
        label_matrix = label_matrix.to(self.adj_parent.device)
        parent_emb = torch.matmul(self.adj_parent, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        # child_emb = torch.matmul(self.adj_child, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        res = torch.matmul(label_matrix, self.w_self) + torch.matmul(parent_emb, self.w_parent) + \
            self.bias
        res = F.relu(res)
        return res


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.attn = nn.Linear(self.out_features * 2, 1, bias=False)
        nn.init.xavier_uniform_(self.attn.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mat):
        n_nodes = h.shape[0]
        g = self.W(h)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        g_repeat = g.repeat(n_nodes, 1) # (N*N, out_features)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0) # (N*N, out_features)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1) # (N*N, 1)
        g_concat = g_concat.view(n_nodes, n_nodes, -1)

        e = self.leakyrelu(self.attn(g_concat)).squeeze(-1) # (N, N)
        assert adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == n_nodes
        assert e.shape[0] == n_nodes and e.shape[1] == n_nodes
        
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        attention = F.softmax(e, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, g)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, adj_parent, adj_child, label_matrix, num_layers=2, hid_q=256, nheads=4, dropout=0, alpha=0.2, encoder_output_size=32, **kwargs):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.register_buffer('label_mat', label_matrix)
        self.register_buffer('adj_parent', adj_parent)
        self.register_buffer('adj_child', adj_child)
        # not heterogeneous
        adj_matrix = adj_parent + adj_child
        one_vec = torch.ones_like(adj_matrix)
        zero_vec = torch.zeros_like(adj_matrix)
        adj_matrix = torch.where(adj_matrix > 0, one_vec, zero_vec)
        self.register_buffer('adj_matrix', adj_matrix)
        
        
        input_dim = label_matrix.shape[1]
        self.dropout = 0

        self.attentions = nn.ModuleList([GraphAttentionLayer(input_dim, hid_q, dropout=self.dropout, alpha=alpha,
                                                             concat=True) for i in range(nheads)])
        self.out_layer = nn.Linear(hid_q * nheads, input_dim)
        self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(input_dim, hid_q, dropout=self.dropout, alpha=alpha,
                                                             concat=True) for i in range(nheads)])
        self.out_layer2 = nn.Linear(hid_q * nheads, encoder_output_size)
        self.saved_matrix = None

    def save_matrix(self, matrix):
        self.saved_matrix = matrix

    def load_matrix(self):
        if self.saved_matrix is None:
            print('no saved matrix, get another one')
            return self.forward(train=True)
        else:
            return self.saved_matrix

    def forward(self, train=True):
        if train is False and self.saved_matrix is not None:
            # print('use saved matrix')
            result = self.load_matrix()
        else:
            x = self.label_mat
            # print('first step', x)
            # x = F.dropout(x, self.dropout, training=self.training)  # type: ignore
            # print('second step', x)
            attn_result = torch.cat([att(x, self.adj_matrix) for att in self.attentions], dim=1)
            # print('third step', attn_result)
            # attn_result = F.dropout(attn_result, self.dropout, training=self.training)
            # print('forth step', attn_result)
            layer1_result_matrix = self.out_layer(attn_result)
            
            attn_result = torch.cat([att(layer1_result_matrix, self.adj_matrix) for att in self.attentions_layer2], dim=1)
            # print('third step', attn_result)
            # attn_result = F.dropout(attn_result, self.dropout, training=self.training)
            # print('forth step', attn_result)
            result = self.out_layer2(attn_result)
            # print('result step', result)
            self.save_matrix(result)
        return result



class Straight(nn.Module):
    def __init__(self, adj_parent, adj_child, label_matrix, num_layers=2, hid_q=256, nheads=4, dropout=0, alpha=0.2, encoder_output_size=32, **kwargs):
        """Dense version of GAT."""
        super(Straight, self).__init__()
        self.register_buffer('label_mat', label_matrix)
        self.register_buffer('adj_parent', adj_parent)
        self.register_buffer('adj_child', adj_child)
        # not heterogeneous
        adj_matrix = adj_parent + adj_child
        one_vec = torch.ones_like(adj_matrix)
        zero_vec = torch.zeros_like(adj_matrix)
        adj_matrix = torch.where(adj_matrix > 0, one_vec, zero_vec)
        self.register_buffer('adj_matrix', adj_matrix)

        self.saved_matrix = self.label_mat

    def save_matrix(self, matrix):
        self.saved_matrix = matrix

    def load_matrix(self):
        if self.saved_matrix is None:
            print('no saved matrix, get another one')
            return self.forward(train=True)
        else:
            return self.saved_matrix

    def forward(self, train=True):
        if train is False and self.saved_matrix is not None:
            # print('use saved matrix')
            result = self.load_matrix()
        else:
            result = self.label_mat
            self.save_matrix(result)
        return result


GraphModel = {
    'GCN': GCN,
    'GCNParent': GCNParent,
    'GAT': GAT,
    'Straight': Straight
}
