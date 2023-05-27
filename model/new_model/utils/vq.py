import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from model.new_model.modules.gcn import GCN, GCNParent
from kmeans_pytorch import kmeans


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.q_name = q_name
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # initialize embedding
        nn.init.xavier_uniform_(self.embeddings.weight)

        self.output_name()

    def output_name(self):
        print(self.q_name, 'Use Hard VQ')

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: meiyong
        '''
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [batch_size, emb_size]
        loss = self.loss_function(quantized, x)

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        result_dict = {
            'quantized': quantized,
            'loss': loss,
            'encoding_indices': encoding_indices
        }
        return result_dict

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight ** 2, dim=1) -
                     2. * torch.matmul(flat_x, self.embeddings.weight.t()))  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,1]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def quantize_embedding(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def loss_function(self, quantized, x):
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # loss = self.commitment_cost * e_latent_loss
        return loss

    def output_similarity(self):
        result_m = self.embeddings.weight/torch.norm(self.embeddings.weight, p=2, dim=1, keepdim=True)
        label_cos = torch.matmul(result_m, result_m.t())
        print('output hard vq similarity', label_cos)

    def reinit_by_kmeans(self, x):
        # kmeans
        print(x)
        cluster_ids_x, cluster_centers = kmeans(
            X=x, num_clusters=self.num_embeddings, distance='euclidean', device=torch.device('cuda:0')
        )
        print(cluster_centers)
        self.embeddings = nn.Embedding.from_pretrained(cluster_centers.to(self.embeddings.weight.device), freeze=False)


class GSVectorQuantizer(nn.Module):
    """
    Gumbel Softmax VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.q_name = q_name
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # initialize embedding
        nn.init.xavier_uniform_(self.embeddings.weight)

        self.output_name()

    def output_name(self):
        print(self.q_name, 'Use gumbel softmax VQ')

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: [batch_size, k]
        '''
        distance = self.get_code_distance(x)  # [batch_size, K]
        if self.training:
            quantized, loss, encoding_indices = self.quantize(distance, var, smooth=True)  # [batch_size, emb_size]
        else:
            quantized, loss, encoding_indices = self.quantize(distance, var, smooth=False)  # [batch_size, emb_size]
            
        result_dict = {
            'quantized': quantized,
            'loss': loss,
            'encoding_indices': encoding_indices
        }
        return result_dict

    def get_code_distance(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [batch_size, K]
        return distances

    def quantize(self, distance, var, smooth=True):
        """
        Returns embedding tensor for a batch of indices.
        distance [batch_size, K]
        var [batch_size, K]
        smooth: use gumbel softmax smoothing
        """
        # print('distance is', distance)
        encoding_indices = torch.argmin(distance, dim=1)  # [N,1]
        dist = RelaxedOneHotCategorical(0.5, logits=-distance)
        # calculate loss
        KL = dist.probs * (dist.logits + math.log(self.num_embeddings))
        KL[dist.probs==0] = 0
        loss = KL.sum(dim=1).mean()
        if smooth:
            sample_probs = dist.rsample()
            quantized = torch.mm(sample_probs, self.embeddings.weight)

        else:
            quantized = self.embeddings(encoding_indices)
        return quantized, loss, encoding_indices

    def quantize_embedding(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def loss_function(self, quantized, x):
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # loss = self.commitment_cost * e_latent_loss
        return loss

    def output_similarity(self):
        result_m = self.embeddings.weight/torch.norm(self.embeddings.weight, p=2, dim=1, keepdim=True)
        label_cos = torch.matmul(result_m, result_m.t())
        print('output hard vq similarity', label_cos)


class DVQ(nn.Module):
    """
    Decomposition vq layer
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, decompose_number, **kwargs):
        super().__init__()
        self.decom_number = decompose_number
        self.embedding_dim = embedding_dim

        self.vq_layers = nn.ModuleList(
            [
                GSVectorQuantizer(num_embeddings, embedding_dim, commitment_cost, q_name=q_name+str(i))
                for i in range(self.decom_number)
            ]
        )

        self.project_layer = nn.Linear(embedding_dim, embedding_dim * self.decom_number)

    def forward(self, inputs, var, **kwargs):
        """
        inputs:
            inputs: torch.tensor 2d, [batch_size, emb_dim]
        outputs:
            quantized: torch.tensor 3d [batch_size, decom_number, emb_dim]
            loss: torch.tensor, sum of loss
            encoding indices: encoding indices
        """
        proj_inputs = self.project_layer(inputs)
        sliced_vecs = proj_inputs.split(self.embedding_dim, dim=1)
        # apply vq to each slice
        vq_out_list = []
        for sliced_vec, vq_layer in zip(sliced_vecs, self.vq_layers):
            vq_out = vq_layer(sliced_vec, var, **kwargs)
            vq_out_list.append(vq_out)
        # aggregate results
        aggregate_out = {}
        keys = vq_out_list[0].keys()
        for k in keys:
            aggregate_out[k] = []
            for vq_out in vq_out_list:
                aggregate_out[k].append(vq_out[k])

        # combine by concatenation
        for i in range(len(aggregate_out["quantized"])):
            aggregate_out["quantized"][i] = aggregate_out["quantized"][i].unsqueeze(1)
        quantized = torch.cat(aggregate_out["quantized"], dim=1)
        # sum losses
        loss = torch.stack(aggregate_out["loss"]).mean()
        # indices
        encoding_indices = torch.stack(aggregate_out["encoding_indices"], dim=-1)

        result_dict = {
            'quantized': quantized,
            'loss': loss,
            'encoding_indices': encoding_indices
        }
        return result_dict
    
    def output_similarity(self):
        for vq_layer in self.vq_layers:
            vq_layer.output_similarity()  # type: ignore


class SoftVQLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, commitment_cost):
        super(SoftVQLoss, self).__init__()
        self.commitment_cost = commitment_cost

    def forward(self, quantized, x):
        '''
        z: [batch, emb_size]
        z_da: [batch, emb_size]
        '''
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = torch.mean((quantized.float() - x.float().detach()) ** 2)
        # commitment loss
        e_latent_loss = torch.mean((quantized.float().detach() - x.float()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print(q_latent_loss, e_latent_loss)
        return loss


class SoftVectorQuantizer(VectorQuantizer):
    """
    Soft VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=None):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, q_name)
        self.loss_function = SoftVQLoss(commitment_cost=commitment_cost)

    def output_name(self):
        print(self.q_name, 'Use Soft VQ')

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: [batch_size, k]
        '''
        distance = self.get_code_distance(x)  # [batch_size, K]
        quantized = self.quantize(distance, var, smooth=smooth)  # [batch_size, emb_size]
        loss = self.loss_function(quantized, x)
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss

    def get_code_distance(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [batch_size, K]
        return distances

    def quantize(self, distance, var, smooth=True):
        """
        Returns embedding tensor for a batch of indices.
        distance [batch_size, K]
        var [batch_size, K]
        """
        # print('distance is', distance)
        if smooth:
            smooth = torch.div(1., torch.exp(var) ** 2)  # smooth var [batch, K]
            prob = torch.exp(-torch.mul(distance, 0.5*smooth))/torch.sqrt(smooth)
            probs = torch.div(prob, torch.sum(prob, dim=1, keepdim=True))  # [batch, K]
        else:
            probs = F.softmax(-distance, dim=1)
        # print('prob max is', torch.max(probs, dim=1))
        # print('prob min is', torch.min(probs, dim=1))
        quantized = torch.mm(probs, self.label_mat)  # [batch, E]
        return quantized, -distance


class FixedVectorQuantizer(nn.Module):
    """
    Fix VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        label_mat: label embedding
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=None):
        super().__init__()
        # label_mat = label_mat/torch.norm(label_mat, p=2, dim=1, keepdim=True)
        self.label_mat = label_mat
        self.q_name = q_name

    def output_name(self):
        print(self.q_name, 'Use Fix Soft VQ')

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: [batch_size, k]
        '''
        distance = self.get_code_distance(x)  # [batch_size, K]
        quantized, new_dis = self.quantize(distance, var, smooth=smooth)  # [batch_size, emb_size]

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, new_dis

    def get_code_distance(self, flat_x):
        # 解决多卡并行device问题
        if self.label_mat.device != flat_x.device:
            self.label_mat = self.label_mat.to(flat_x.device)
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.label_mat ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.label_mat.t())
        )  # [batch_size, K]
        return distances

    def quantize(self, distance, var, smooth=True):
        """
        Returns embedding tensor for a batch of indices.
        distance [batch_size, K]
        var [batch_size, K]
        """
        # print('distance is', distance)
        if smooth:
            smooth = torch.div(1., torch.exp(var) ** 2)  # smooth var [batch, K]
            prob = torch.exp(-torch.mul(distance, 0.5*smooth))/torch.sqrt(smooth)
            probs = torch.div(prob, torch.sum(prob, dim=1, keepdim=True))  # [batch, K]
        else:
            probs = F.softmax(-distance, dim=1)
        # print('prob max is', torch.max(probs, dim=1))
        # print('prob min is', torch.min(probs, dim=1))

        encoding_indices = torch.argmin(distance, dim=1)  # [N,1]
        return self.label_mat[encoding_indices], -distance

    def quantize_embedding(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.label_mat[encoding_indices]


class FixedVectorSoftQuantizer(FixedVectorQuantizer):
    """
    Fix VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        label_mat: label embedding
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=None):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=None)

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: [batch_size, k]
        '''
        distance = self.get_code_distance(x)  # [batch_size, K]
        quantized, new_dis = self.quantize(distance, var, smooth=smooth)  # [batch_size, emb_size]

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, new_dis

    def quantize(self, distance, var, smooth=True):
        """
        Returns embedding tensor for a batch of indices.
        distance [batch_size, K]
        var [batch_size, K]
        """
        print('distance is', distance)
        if smooth:
            smooth = torch.div(1., torch.exp(var) ** 2)  # smooth var [batch, K]
            print(smooth)
            prob = torch.exp(-torch.mul(distance, 0.5*smooth))/torch.sqrt(smooth)
            print(prob)
            probs = torch.div(prob, torch.sum(prob, dim=1, keepdim=True))  # [batch, K]
        else:
            probs = F.softmax(-distance, dim=1)
        print('prob max is', torch.max(probs, dim=1))
        print('prob min is', torch.min(probs, dim=1))
        quantized = torch.mm(probs, self.label_mat)  # [batch, E]
        return quantized, -distance


class FixedVectorQuantizerGCN(FixedVectorQuantizer):
    """
    Fix VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        label_mat: label embedding
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=None,
                 adj_parent=None, adj_child=None):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, q_name, label_mat=label_mat)
        # two gcn layer
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        print('in vq, adj is', adj_parent, adj_child)
        self.emb_size = label_mat.shape[1]
        self.first_gcn = GCNParent(adj_child=self.adj_child, adj_parent=self.adj_parent, input_dim=self.emb_size,
                                   hid_q=self.emb_size)
        self.sceond_gcn = GCNParent(adj_child=self.adj_child, adj_parent=self.adj_parent, input_dim=self.emb_size,
                                    hid_q=self.emb_size)

    def output_name(self):
        print(self.q_name, 'Use Fix GCN VQ')

    def gcn(self, label_mat):
        label_mat_1 = self.first_gcn(label_mat)  # [label_num, hid_q]
        label_mat_2 = self.sceond_gcn(label_mat_1)  # [label_num, hid_q]
        return label_mat_2

    def forward(self, x, var, smooth=True):
        '''
        x: [batch_size, emb_size]
        var: [batch_size, k]
        '''
        label_mat = self.gcn(self.label_mat)
        distance = self.get_code_distance(x, label_mat)  # [batch_size, K]
        quantized, new_dis = self.quantize(distance, var, label_mat, smooth=smooth)  # [batch_size, emb_size]

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, new_dis

    def get_code_distance(self, flat_x, label_mat):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(label_mat ** 2, dim=1) -
            2. * torch.matmul(flat_x, label_mat.t())
        )  # [batch_size, K]
        return distances

    def quantize(self, distance, var, label_mat, smooth=True):
        """
        Returns embedding tensor for a batch of indices.
        distance [batch_size, K]
        var [batch_size, K]
        """
        # print('distance is', distance)
        if smooth:
            smooth = torch.div(1., torch.exp(var) ** 2)  # smooth var [batch, K]
            prob = torch.exp(-torch.mul(distance, 0.5*smooth))/torch.sqrt(smooth)
            probs = torch.div(prob, torch.sum(prob, dim=1, keepdim=True))  # [batch, K]
        else:
            probs = F.softmax(-distance, dim=1)

        encoding_indices = torch.argmin(distance, dim=1)  # [N,1]
        return self.quantize_embedding(encoding_indices, label_mat), -distance

    def quantize_embedding(self, encoding_indices, label_mat=None):
        """Returns embedding tensor for a batch of indices."""
        if label_mat is None:
            label_mat = self.gcn(self.label_mat)
        return label_mat[encoding_indices]


class VAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, q_name, **kwargs):
        super().__init__()
        pass

    def forward(self, x, var, **kwargs):
        '''
        vae forward
        '''
        z = self.reparameterize(x, var)
        loss = self.kl_loss(x, var)
        encoding_indices = torch.zeros(z.shape[0]).to(z.device)
        result_dict = {
            'quantized': z,
            'loss': loss,
            'encoding_indices': encoding_indices
        }
        return result_dict 

    def kl_loss(self, mu, logvar):
        loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return loss.mean()

    def reparameterize(self, mu, logvar):
        '''
        reparameterize trick
        '''
        std = torch.exp(0.5 * logvar).to(mu.device)
        eps = torch.randn_like(std).to(mu.device)
        return mu + eps * std

    def output_similarity(self):
        pass

# VQ type
VQ = {
    'Soft': SoftVectorQuantizer,
    'Hard': VectorQuantizer,
    'GSHard': GSVectorQuantizer,
    'Fix': FixedVectorQuantizer,
    'Fix_Soft': FixedVectorSoftQuantizer,
    'Fix_GCN': FixedVectorQuantizerGCN,
    'DVQ': DVQ,
    'VAE': VAE
}
