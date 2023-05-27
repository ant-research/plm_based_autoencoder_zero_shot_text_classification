import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    CNN encoder of the input sentence
    '''
    def __init__(self, word_emb_len: int, filter_sizes: list = [10], out_dim: int = 50):
        """
        Args:
             (2d tensor): pretrained word embedding matrix
        """
        super(Encoder, self).__init__()
        self.word_emb_len = word_emb_len

        # the convolution layer for the word embedding
        self.filter = nn.Conv2d(in_channels=1, out_channels=out_dim,
                                kernel_size=(filter_sizes[0], self.word_emb_len), bias=True)
        nn.init.kaiming_normal_(self.filter.weight)
        nn.init.constant_(self.filter.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        D = self.filter(x)  # [batch_size, word_num - filter_width + 1, filter_num]
        D = D.squeeze(-1)
        D = D.permute(0, 2, 1)
        return D


class GCN(nn.Module):
    def __init__(self, adj_parent, adj_child, input_dim, hid_q=256):
        """
        RGCN
        Args:
            adj_parent (2d tensor): adjacency matrix of parent
            adj_child (2d tensor): adjacency matrix of child
            input_dim (int): intput dimension
            hid_q (int): output dimension
        """
        super(GCN, self).__init__()
        self.adj_parent = adj_parent
        self.adj_child = adj_child

        self.w_self = nn.Parameter(torch.Tensor(input_dim, hid_q))
        self.w_parent = nn.Parameter(torch.Tensor(input_dim, hid_q))
        self.w_child = nn.Parameter(torch.Tensor(input_dim, hid_q))
        self.bias = nn.Parameter(torch.Tensor(hid_q))
        nn.init.kaiming_normal_(self.w_self)
        nn.init.kaiming_normal_(self.w_parent)
        nn.init.kaiming_normal_(self.w_child)
        nn.init.constant_(self.bias, 0)

    def forward(self, label_matrix):
        """
        Args:
            label_matrix (2d tensor): label embedding matrix
        Returns:
            new_label_matrix (2d tensor): new label embedding matrix
        """
        parent_emb = torch.matmul(self.adj_parent, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        child_emb = torch.matmul(self.adj_child, label_matrix)  # [label_num, label_num] * [label_num, word_emb]
        res = torch.matmul(label_matrix, self.w_self) + torch.matmul(parent_emb, self.w_parent) + torch.matmul(child_emb, self.w_child) + self.bias
        res = F.relu(res)
        return res


class MLZS(nn.Module):
    ''' CNN Model (http://www.aclweb.org/anthology/D14-1181)
    '''
    def __init__(self, label_mat, adj_parent, adj_child, fs=[10], nf=50, hid_q=256):
        """
        Args:
            adj_parent (2d tensor): adjacency matrix of parent
            adj_child (2d tensor): adjacency matrix of child
            label_mat (2d tensor [label_num, word_emb]): label embedding matrix
            hid_q (int): dimension of gcn
        """
        super(MLZS, self).__init__()
        self.label_mat = label_mat
        self.adj_parent = F.softmax(adj_parent, dim=1)  # [label_num, label_num]
        self.adj_child = F.softmax(adj_child, dim=1)  # [label_num, label_num]
        self.hid_q = hid_q
        self.word_emb_len = label_mat.shape[1]

        # start the convolution layer for the word embedding
        self.encoder = Encoder(word_emb_len=self.word_emb_len, filter_sizes=fs, out_dim=nf)

        # start the label wise attention layer
        # get the D square
        self.square_layer = nn.Linear(in_features=nf*len(fs), out_features=self.word_emb_len, bias=True)
        nn.init.kaiming_normal_(self.square_layer.weight)
        nn.init.constant_(self.square_layer.bias, 0)
        # get the dimension match layer
        self.dim_match_layer = nn.Linear(in_features=nf*len(fs),
                                         out_features=(self.hid_q + self.word_emb_len), bias=True)
        nn.init.kaiming_normal_(self.dim_match_layer.weight)
        nn.init.constant_(self.dim_match_layer.bias, 0)

        # two gcn layer
        self.first_gcn = GCN(adj_child=self.adj_child, adj_parent=self.adj_parent, input_dim=self.word_emb_len,
                             hid_q=self.hid_q)
        self.sceond_gcn = GCN(adj_child=self.adj_child, adj_parent=self.adj_parent, input_dim=self.hid_q,
                              hid_q=self.hid_q)

    def attention_net(self, q, k, v):
        """
        Args:
            q (3d tensor): query
            k (3d tensor): key
            v (3d tensor): value
        Return:
            res (3d tensor): attenion result
        """
        att_weight = torch.matmul(q, k)  # [batch_size, word_num - filter_width + 1, word_emb] * [word_emb, label_num]
        softmax_att_weight = F.softmax(att_weight, dim=1)  # [batch_size, word_num - filter_width + 1, label_num]
        softmax_att_weight = softmax_att_weight.permute(0, 2, 1)  # [batch_size, label_num, word_num - filter_width + 1]
        res = torch.matmul(softmax_att_weight, v)
        # [batch_size, label_num, word_num - filter_width + 1] * [batch_size, word_num - filter_width + 1, filter_num]
        return res

    def forward(self, x):
        # embedding D
        D = self.encoder(x)  # [batch_size, word_num - filter_width + 1, filter_num]

        # get D square = tanh(linear(D))
        D_square = self.square_layer(D)  # [batch_size, word_num - filter_width + 1, word_emb]
        # get attention result
        label_emb = self.label_mat.transpose(1, 0)
        c_att = self.attention_net(D_square, label_emb, D)  # [batch_size, label_num, filter_num]
        e_att = F.relu(self.dim_match_layer(c_att))  # [batch_size, label_num, hid_q + word_emb]

        # get gcn output
        label_mat_1 = self.first_gcn(self.label_mat)  # [label_num, hid_q]
        label_mat_2 = self.sceond_gcn(label_mat_1)  # [label_num, hid_q]
        # concat label embedding
        label_mat_3 = torch.cat((self.label_mat, label_mat_2), dim=1)  # [label_num, hid_q + word_emb]
        # get result
        e_att_T = e_att.transpose(2, 1)
        print('eattT shape', e_att_T.shape)
        y_mat = torch.matmul(label_mat_3, e_att_T)
        print(y_mat.shape)
        res = torch.diagonal(y_mat, dim1=-2, dim2=-1)
        # res = F.sigmoid(res)
        # res = res.transpose(1, 0)
        print(res.shape)
        # print(res.shape)
        return res