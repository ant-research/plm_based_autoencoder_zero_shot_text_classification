import math
from turtle import distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from model.discrete_vae.models.simple_module import PackedSequneceUtil


class FixedVectorQuantizer(nn.Module):
    def __init__(
        self, config, num_embeddings, label_mat, **kwargs
    ):
        super(FixedVectorQuantizer, self).__init__()
        """
        label mat: 初始化的label matrix，可以选择在后续通过输入的gnn结果更新或者一直fix住
        """
        self.saved_label_matrix = label_mat
        self._num_embeddings = num_embeddings # 有多少个label
        self.commitment_cost = config.comit_coef
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, label_matrix, y=None):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        Returns:
            quantize: Tensor containing the quantized version of the input.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
        inputs: B × T (optional) × D
        """
        # save new label matrix
        self.saved_label_matrix = label_matrix

        # l2 distances between z_e and embedding vectors: (bsz * decompose) × K
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.saved_label_matrix ** 2, dim=1)
            - 2 * torch.matmul(inputs, self.saved_label_matrix.t())
        )

        """
        encoding_indices: Tensor containing the discrete encoding indices, i.e.
        which element of the quantized space each input element was mapped to.
        """
        # encoding_indices: bsz * decompose
        min_distances, encoding_indices = torch.min(distances, dim=1)
        # print(self.training, y, y is None)
        if (self.training) and ((y is None) == False):
            print('have label y')
            loss = self.loss_func(-distances, y)
        else:
            print('no_label_Y_input')
            loss = self.loss_func(-distances, encoding_indices)
 

        # Quantize and unflatten
        quantized = self.saved_label_matrix[encoding_indices]
        loss_comit = F.mse_loss(inputs, quantized.detach(), reduction="sum")  * self.commitment_cost
        
        # straight through gradient
        quantized_st = inputs + (quantized - inputs).detach()

        output_dict = {
            # B × T (optional) × (M * D)
            "quantized": quantized_st,
            # B × T (optional) × M × D
            "quantized_stack": quantized_st.unsqueeze(1),
            # B × T (optional) × M
            "encoding_indices": encoding_indices,
            "loss": loss_comit,
            "loss_commit": loss_comit.detach(),
            "min_distances": min_distances,
            "distances": distances,
            'classifier_loss': loss
        }
        return output_dict

    def quantize_embedding(self, indice: torch.Tensor) -> torch.Tensor:
        return self.saved_label_matrix[indice]


class FixedVectorQuantizerClassifier(nn.Module):
    def __init__(
        self, config, num_embeddings, label_mat, **kwargs
    ):
        super(FixedVectorQuantizerClassifier, self).__init__()
        """
        label mat: 初始化的label matrix，可以选择在后续通过输入的gnn结果更新或者一直fix住
        """
        self.saved_label_matrix = label_mat
        self._num_embeddings = num_embeddings # 有多少个label
        self.commitment_cost = config.comit_coef
        
        self.classifier_layer1 = nn.Linear(self.saved_label_matrix.shape[1]*2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, label_matrix, y=None):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        Returns:
            quantize: Tensor containing the quantized version of the input.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
        inputs: B × T (optional) × D
        """
        # save new label matrix
        self.saved_label_matrix = label_matrix

        # l2 distances between z_e and embedding vectors: (bsz * decompose) × K
        label_mat = self.saved_label_matrix.unsqueeze(0).repeat(inputs.shape[0], 1, 1).to(inputs.device)
        cls_num = label_mat.shape[1]  # label count
        z_t_mat = inputs.unsqueeze(1).repeat(1, cls_num, 1)  # [batch, label_count, distil_size]
        # print('shape is:', z_t_mat.shape, label_mat.shape, inputs.shape, label_matrix.shape)
        relation_pairs = torch.cat([z_t_mat, label_mat], 2)
        prob = self.classifier_layer1(relation_pairs)  # [batch * label_count, 256]
        prob = self.fc2(self.relu(prob))
        prob = prob.reshape(inputs.shape[0], -1)


        """
        encoding_indices: Tensor containing the discrete encoding indices, i.e.
        which element of the quantized space each input element was mapped to.
        """
        # encoding_indices: bsz * decompose
        min_distances, encoding_indices = torch.max(prob, dim=1)
        
        if y is not None:
            loss = self.loss_func(prob, y)
        else:
            loss = self.loss_func(prob, encoding_indices)

        # Quantize and unflatten
        quantized = self.saved_label_matrix[encoding_indices]
        
        loss_comit = F.mse_loss(inputs, quantized.detach(), reduction="sum")  * self.commitment_cost
        # straight through gradient
        quantized_st = inputs + (quantized - inputs).detach()

        output_dict = {
            # B × T (optional) × (M * D)
            "quantized": quantized_st,
            # B × T (optional) × M × D
            "quantized_stack": quantized_st.unsqueeze(1),
            # B × T (optional) × M
            "encoding_indices": encoding_indices,
            "loss": loss_comit,
            "loss_commit": loss_comit.detach(),
            "min_distances": min_distances,
            "distances": -prob,
            'classifier_loss': loss
        }
        return output_dict

    def quantize_embedding(self, indice: torch.Tensor) -> torch.Tensor:
        return self.saved_label_matrix[indice]