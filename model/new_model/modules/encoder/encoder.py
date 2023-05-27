import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from model.new_model.modules.simple_modules import PositionalEncoding, Pooler


class Transformer(nn.Module):
    '''
    Transformer的encoder部分
    '''
    def __init__(self, config, device, output_size, gpu_tracker, encoder_name):
        super(Transformer, self).__init__()
        self.gpu_tracker = gpu_tracker
        if config.emb_size % config.num_heads != 0:
            raise ValueError("Embedding size %d needs to be divisible by number of heads %d"
                             % (config.emb_size, config.num_heads))
        self.emb_size = config.emb_size
        self.device = device
        self.k = config.class_num
        self.hid_emb = config.emb_size
        self.max_len = config.seq_len
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        self.output_size = output_size
        self.encoder_name = encoder_name
        # CLS vector
        self.cls = torch.rand(self.emb_size) 
        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))

        # Transformer dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # Transformer positional encoder
        self.pos_encoder = PositionalEncoding(self.emb_size, self.dropout_rate, self.max_len+1)
        # Transformer Encoder Layer
        self.input_layer = nn.Linear(self.emb_size, self.hid_emb, bias=True)
        encoder_layers = TransformerEncoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                 dim_feedforward=self.forward_expansion*self.hid_emb,
                                                 dropout=self.dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, self.num_layers)
        self.output_layer = nn.Linear(self.hid_emb, self.output_size)
        self.var_layer = nn.Linear(self.hid_emb, self.output_size)

    def forward(self, inputs, pad_mask=None, targets=None):
        '''
        inputs: [N, S, E]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        #  positions with the value of True will be ignored while the position with the value of False will be unchanged
        if pad_mask is None:
            src_key_padding_mask = None
        else:
            cls_pad = torch.ones([inputs.shape[0], 1]).to(pad_mask.device)
            pad_mask = torch.cat([cls_pad.float(), pad_mask], dim=1)
            src_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)            

        # add cls token and padding_mask
        cls_matrix = self.cls.repeat(inputs.shape[0], 1, 1).to(inputs.device)  # expand to [batch_size, 1, emb_size]
        inputs = torch.cat([cls_matrix, inputs], dim=1)  # [batch_size, seq_len+1, emb_size]
        # add positional encoding
        inputs = inputs.permute(1, 0, 2)  # [S, N, E]
        # embeds = self.pos_encoder(inputs * math.sqrt(self.emb_size))  # [S, N, E]
        embeds = self.pos_encoder(inputs)  # [S, N, E]
        embeds = self.dropout(embeds)
        # 减少参数层
        embeds = self.input_layer(embeds)
        # embeds = checkpoint(self.encoder, embeds, None, src_key_padding_mask)  # [T, N, E]
        embeds = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        # embeds = self.encoder(embeds)
        #self.gpu_tracker.track()
        encoded_inputs = embeds.permute(1, 0, 2)  # [N, T, E]

        # get cls output
        output = encoded_inputs[:, 0, :].squeeze(1)
        result = self.output_layer(output)  # [N, output_size]
        var = self.var_layer(output)  # [N, K]
        return result, var


class Transformer_Disperse(Transformer):
    '''
    Transformer的encoder部分
    '''
    def __init__(self, config, device, output_size, gpu_tracker, encoder_name):
        super().__init__(config, device, output_size, gpu_tracker, encoder_name)
        self.conv_layer = nn.Linear(self.max_len+1, 1)

    def forward(self, inputs, pad_mask=None, targets=None):
        '''
        inputs: [N, S, E]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if pad_mask is None:
            src_key_padding_mask = None
        else:
            cls_pad = torch.ones([inputs.shape[0], 1]).to(pad_mask.device)
            pad_mask = torch.cat([cls_pad.float(), pad_mask], dim=1)
            src_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)            

        # add cls token and padding_mask
        cls_matrix = self.cls.repeat(inputs.shape[0], 1, 1).to(inputs.device)  # expand to [batch_size, 1, emb_size]
        inputs = torch.cat([cls_matrix, inputs], dim=1)  # [batch_size, seq_len+1, emb_size]
        # add positional encoding
        inputs = inputs.permute(1, 0, 2)  # [S, N, E]
        # embeds = self.pos_encoder(inputs * math.sqrt(self.emb_size))  # [S, N, E]
        embeds = self.pos_encoder(inputs)  # [S, N, E]
        embeds = self.dropout(embeds)
        # 减少参数层
        embeds = self.input_layer(embeds)
        # embeds = checkpoint(self.encoder, embeds, None, src_key_padding_mask)  # [T, N, E]
        embeds = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        # embeds = self.encoder(embeds)
        encoded_inputs = embeds.permute(1, 0, 2)  # [N, T, E]

        # get cls output
        encoded_inputs = encoded_inputs.permute(0, 2, 1)
        output = self.conv_layer(encoded_inputs).squeeze(2)
        result = self.output_layer(output)  # [N, output_size]
        var = self.var_layer(output)  # [N, K]
        return result, var


class TransformerQuantizerEncoder(nn.Module):
    """transformer encoder + vq

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config, emb_layer, encoder_name, vq_type, **kwargs):
        super().__init__()
        self.use_memory = config.use_memory
        self.hid_emb = config.emb_size
        self.output_dim = config.encoder_output_size
        self.dropout = config.dropout_rate
        self.num_heads = config.num_heads
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        self.vq_type = vq_type
        
        self.pad_idx = emb_layer.emb_model.pad_idx
        self.num_embeddings = emb_layer.emb_model.get_vocab_len()
        self.max_len = config.seq_len
        
        print('%s Transformer setting: emb size %d, max len %d' % (encoder_name, self.hid_emb, self.max_len))

        if config.use_w2v_weight is False:
            self.embedding = nn.Embedding(
                self.num_embeddings, self.hid_emb, padding_idx=self.pad_idx
            )
        else:
            print('use pretrained weight')
            # self.emb_model = nn.Embedding.from_pretrained(self.emb_layer.emb_model.embedding_layer.weight, freeze=True)

            self.embedding = emb_layer.emb_model.emb_from_idx

        self.pos_encoder = PositionalEncoding(
            d_model=self.hid_emb, dropout=self.dropout, max_len=self.max_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_emb,
            nhead=self.num_heads,
            dim_feedforward=self.forward_expansion,
            dropout=self.dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        
        # sentence
        if self.vq_type in ["Striaght", "DVQ", "Fix"]:
            d_proj = config.decompose_number * self.output_dim  # output dimension * decompose number
        elif self.vq_type in ['GS']:
            d_proj = config.decompose_number * config.disper_num
        else:
            raise KeyError
        
        self.pooler = Pooler(
            project=True,
            pool_type="mean",
            d_inp=self.hid_emb,
            d_proj=d_proj,
        )

    def forward(self, src):
        src_pad_mask = src == self.pad_idx
        src_nopad_mask = src != self.pad_idx

        src_emb = self.embedding(src).transpose(0, 1) # [N, B, E]
        src_emb = self.pos_encoder(src_emb)
        src_mask = None

        memory = self.encoder(
            src_emb, src_key_padding_mask=src_pad_mask, mask=src_mask
        ).transpose(0, 1) # [B, N, E]

        # seq2vec: mean pooling to get sentence representation
        pooled_memory = self.pooler(memory, src_nopad_mask)  # [B, E*decompose_number]
        print('pooled memory is', pooled_memory)
        
        return {
            "pooled_memory": pooled_memory,
            "nopad_mask": src_nopad_mask
        }

    def get_output_dim(self):
        return self.output_dim
