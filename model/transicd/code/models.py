import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Attention(nn.Module):
    def __init__(self, hidden_size, output_size, attn_expansion, dropout_rate):
        super(Attention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size*attn_expansion)
        self.tnh = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size*attn_expansion, output_size)

    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x H -> B x S x attn_expansion*H
        output_1 = self.tnh(self.l1(hidden))
        # output_1 = self.dropout(output_1)

        # output_2: B x S x attn_expansion*H -> B x S x output_size(O)
        output_2 = self.l2(output_1)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x output_size(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class LabelAttention(nn.Module):
    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(LabelAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # output_1: B x S x H -> B x S x E
        output_1 = self.tnh(self.l1(hidden))
        output_1 = self.dropout(output_1)
        # output_2: (B x S x E) x (E x L) -> B x S x L
        output_2 = torch.matmul(output_1, label_embeds.t().to(output_1.device))

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x L -> B x L x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len, num_layers, num_heads, forward_expansion,
                 output_size, dropout_rate, device, pad_idx=0):
        super(Transformer, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*embed_size, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])

    def forward(self, inputs, targets=None):
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S

        embeds = self.pos_encoder(self.embedder(inputs) * math.sqrt(self.embed_size))  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
        encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E

        pooled_outputs = encoded_inputs.mean(dim=1)
        outputs = torch.zeros((pooled_outputs.size(0), self.output_size)).to(self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(pooled_outputs)

        return outputs, None, None


class TransICD(nn.Module):
    def __init__(self, config, label_matrix, device, gpu_tracker, label_freq=None, C=3.0):
        super(TransICD, self).__init__()

        # model config
        self.embed_size = config.emb_size
        self.max_len = config.seq_len
        self.class_num = config.class_num
        self.device = device
        self.pad_idx = config.pad_idx
        self.label_mat = label_matrix

        # transformer config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.forward_expansion = config.forward_expansion
        self.dropout_rate = config.dropout_rate

        # attenion config
        self.attn_expansion = config.attn_expansion

        if self.embed_size % self.num_heads != 0:
            raise ValueError(f"Embedding size {self.embed_size} needs to be divisible by number of heads {self.num_heads}")

        if label_freq is not None:
            class_margin = torch.tensor(label_freq, dtype=torch.float32) ** 0.25
            class_margin = class_margin.masked_fill(class_margin == 0, 1)
            self.register_buffer('class_margin', 1.0 / class_margin)
            self.C = C
        else:
            self.class_margin = None
            self.C = 0

        self.dropout = nn.Dropout(self.dropout_rate)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout_rate, self.max_len)
        encoder_layers = TransformerEncoderLayer(d_model=self.embed_size, nhead=self.num_heads,
                                                 dim_feedforward=self.forward_expansion*self.embed_size, dropout=self.dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, self.num_layers)
        # self.attn = Attention(self.embed_size, self.class_num, self.attn_expansion, self.dropout_rate)
        self.label_attn = LabelAttention(self.embed_size, self.embed_size, self.dropout_rate)
        self.fcs = nn.ModuleList([nn.Linear(self.embed_size, 1) for code in range(self.class_num)])

    def forward(self, inputs, targets=None, input_idx=None):
        if input_idx:
            # attn_mask: B x S -> B x S x 1
            attn_mask = (input_idx != self.pad_idx).unsqueeze(2).to(self.device)
        else:
            attn_mask = None

        embeds = self.pos_encoder(inputs * math.sqrt(self.embed_size))  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E
        embeds = checkpoint(self.encoder, embeds)  # [T, N, E]
        encoded_inputs = embeds.permute(1, 0, 2)  # N x T x E

        # encoded_inputs is of shape: batch_size, seq_len, embed_size
        # weighted_outputs, attn_weights = self.attn(encoded_inputs, attn_mask)
        # label_embeds = self.embed_label_desc()
        weighted_outputs, attn_weights = self.label_attn(encoded_inputs, self.label_mat, attn_mask)

        outputs = torch.zeros((weighted_outputs.size(0), self.class_num)).to(self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(weighted_outputs[:, code, :])

        if targets is not None and self.class_margin is not None and self.C > 0:
            ldam_outputs = outputs - targets * self.class_margin * self.C
        else:
            ldam_outputs = None

        return outputs, ldam_outputs, attn_weights
