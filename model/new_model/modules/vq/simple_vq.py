import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightForwardZ(nn.Module):
    """
    Continuous relaxation of categorical distrubution (the two are equivalent)
        Concrete distribution: https://arxiv.org/pdf/1611.00712.pdf
        Gumbel-Softmax distribution: https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_embeddings, embedding_dim, config, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.split = config.decompose_number

    def forward(self, inputs, **kwargs):
        pooled_memory = inputs
        stacked_pooled_memory = pooled_memory.reshape(pooled_memory.shape[0], -1, self.embedding_dim)
        assert stacked_pooled_memory.shape[1] == self.split, 'memory size unmatched'
        encoding_indices = torch.zeros(pooled_memory.shape[0])
        return {
            # B x T (optional) x (M * D)
            "quantized": pooled_memory,
            # B x T (optional) x M x D
            "quantized_stack": stacked_pooled_memory,
            # B x T (optional) x M
            "encoding_indices": encoding_indices,
        }
    
    def quantize_embedding(self, indice: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError