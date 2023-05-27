import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

def one_hot_argmax(y_soft, dim=-1):
    """
    Example:
    y_soft = [0.5, 0.2, 0.3] # logits vector (normalized or unnormalized)
    y_hard = [1., 0, 0]      # one-hot vector for argmax
    """
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    return y_hard


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)  # type: ignore
        return self.dropout(x)


class PackedSequneceUtil(object):
    def __init__(self):
        self.is_packed = False
        self.pack_shape = None

    def preprocess(self, input):
        self.is_packed = isinstance(input, PackedSequence)
        if self.is_packed:
            input, *self.pack_shape = input
        return input

    def postprocess(self, output, pad):
        assert self.is_packed
        packed_ouput = PackedSequence(output, *self.pack_shape)  # type: ignore
        padded_output = pad_packed_sequence(
            packed_ouput, batch_first=True, padding_value=pad
        )[0]
        return padded_output


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, project=True, d_inp=512, d_proj=512, pool_type="max"):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

        if self.pool_type == "attn":
            d_in = d_proj if project else d_inp
            self.attn = nn.Linear(d_in, 1, bias=False)

    def forward(self, sequence, mask):
        """
        sequence: (bsz, T, d_inp)
        mask: nopad_mask (bsz, T) or (bsz, T, 1)
        """
        # no pad in sequence
        if mask is None:
            mask = torch.ones(sequence.shape[:2], device=sequence.device)

        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)  # (bsz, T, 1)
        pad_mask = mask == 0
        proj_seq = self.project(sequence)  # (bsz, T, d_proj) or (bsz, T, d_inp)

        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]

        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1).float()

        elif self.pool_type == "final":
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs).squeeze(dim=1)

        elif self.pool_type == "first":
            seq_emb = proj_seq[:, 0]

        elif self.pool_type == "none":
            seq_emb = proj_seq
        else:
            raise KeyError
        
        return seq_emb

    def forward_dict(self, output_dict):
        """
        Arg - output_dict with keys:
            'output': sequence of vectors
            'nopad_mask': sequence mask, with 1 for non pad positions and 0 elsewhere
            'final_state' (optional): final hidden state of lstm
        Return - an aggregated vector
        """
        sequence = output_dict["sequence"]
        mask = output_dict["nopad_mask"]

        if self.pool_type == "final_state":
            assert "final_state" in output_dict
            out = output_dict["final_state"]
        else:
            out = self.forward(sequence, mask)
        return out