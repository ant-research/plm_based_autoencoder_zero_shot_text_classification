"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import copy
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from model.new_model.modules.simple_modules import PositionalEncoding


class GPTLayer(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, hid_emb, num_heads, dropout_rate, forward_expansion):
        super().__init__()
        self.hid_emb = hid_emb
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forward_expansion = forward_expansion

        self.ln1 = nn.LayerNorm(self.hid_emb)
        self.ln2 = nn.LayerNorm(self.hid_emb)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        # MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=self.hid_emb,
                                          num_heads=self.num_heads,
                                          dropout=self.dropout_rate)
        # FeedForward
        self.mlp = nn.Sequential(
            nn.Linear(self.hid_emb, self.forward_expansion * self.hid_emb),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.forward_expansion * self.hid_emb, self.hid_emb),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        """
        x: torch.Tensor [seq_len, batch_size, hid_emb]
        """
        # self multihead attention
        x = x + self._sa_block(self.ln1(x), attn_mask, key_padding_mask)
        # output layer
        x = x + self._ff_block(self.ln2(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # self multihead attention block
        x = self.attn(x, x, x,
                      attn_mask=attn_mask,
                      key_padding_mask=key_padding_mask,
                      need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.mlp(x)
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GPTDecoder(nn.Module):
    """GPT layer"""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, z: torch.Tensor, attn_mask: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, z_fix: bool = False) -> torch.Tensor:
        """
        x: [seq_len, batch_size, hid_emb]
        z: [z_len, batch_size, hid_emb]
        attn_mask: [batch_size, seq_len + z_len]
        key_padding_mask: [batch_size, seq_len + z_len]
        z_fix: bool fix z after one layer or not
        """
        output = torch.cat([z, x], dim=0)

        for gpt_layer in self.layers:
            output = gpt_layer(output, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
            if z_fix:
                z_len = z.shape[0]
                output = torch.cat([z, output[z_len:, :, :]], dim=0)
            else:
                pass

        return output

    def pretrain(self, x: torch.Tensor, attn_mask: torch.Tensor,
                 key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [seq_len, batch_size, hid_emb]
        z: [z_len, batch_size, hid_emb]
        attn_mask: [batch_size, seq_len + z_len]
        key_padding_mask: [batch_size, seq_len + z_len]
        z_fix: bool fix z after one layer or not
        """
        output = x

        for gpt_layer in self.layers:
            output = gpt_layer(output, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)

        return output


class GPTModelBase(nn.Module):
    """
    Base model for gpt idx version and gpt embedding version
    """
    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = True, z_fix: bool = True,
                 only_t: bool = False):
        super().__init__()
        # set config
        self.device = device
        self.hid_emb = config.emb_size
        self.emb_size = config.emb_size
        self.dropout_rate = config.dropout_rate
        self.max_len = config.seq_len
        self.only_t = only_t  # 前面只用z_t或z_t + z_p

        # gpt self_attention config
        self.forward_expansion = config.forward_expansion
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.z_fix = z_fix

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_bos_prob(self, shape):
        """获取bos的概率"""
        bos_prob = torch.zeros([shape, 1, self.n_tokens])
        bos_prob[:, :, self.bos_idx] = 1e8
        return bos_prob

    def generate_square_subsequent_mask(self, size):
        """生成mask"""
        # Generate mask covering the top right triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class GPT(GPTModelBase):
    """
    the full GPT language model, with a context size of block_size
    decoder_new_embedding: Build a new embedding layer or use the trained one
    decoder_z_fix: fix z in the GPT training or not
    decoder_only_t: use only z_t as the head of the gpt or use z_t + z_p
    """

    def __init__(self, config, device, bos, emb_layer, decoder_new_embedding: bool = True, decoder_z_fix: bool = True,
                 decoder_only_t: bool = True, **kwargs):
        super().__init__(config, device, bos, emb_layer, decoder_new_embedding,
                         z_fix=decoder_z_fix, only_t=decoder_only_t, **kwargs)
        # vocab setting
        self.emb_layer = emb_layer
        self.bos_idx = self.emb_layer.emb_model.bos_idx
        self.eos_idx = self.emb_layer.emb_model.eos_idx
        self.mask_idx = self.emb_layer.emb_model.mask_idx
        self.n_tokens = self.emb_layer.emb_model.get_vocab_len()

        # input embedding stem
        if decoder_new_embedding:
            self.emb_model = nn.Embedding(self.n_tokens, self.hid_emb)
        else:
            print('use pretrained weight')
            # self.emb_model = nn.Embedding.from_pretrained(self.emb_layer.emb_model.embedding_layer.weight, freeze=True)
            self.emb_model = self.emb_layer.emb_model.emb_from_idx
        if self.only_t:
            self.pos_encoder = PositionalEncoding(self.hid_emb, self.dropout_rate, self.max_len + 10)
        else:
            self.pos_encoder = PositionalEncoding(self.hid_emb, self.dropout_rate, self.max_len + 10)
        self.drop = nn.Dropout(self.dropout_rate)
        # GPT
        gpt_layer = GPTLayer(self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)
        # decoder head
        self.ln_f = nn.LayerNorm(self.hid_emb)
        self.output_layer = nn.Linear(self.hid_emb, self.n_tokens, bias=False)
        # z memory and emb layer
        self.z_embedding_layer = nn.Linear(config.distil_size, self.hid_emb)
        self.z_memory_layer = nn.Linear(config.disper_size, self.hid_emb)
        # emb_layer
        concat_emb = 3 * self.hid_emb
        self.small_hid_emb_layer = nn.Linear(concat_emb, self.hid_emb)

        self._reset_parameters()

    def deal_z(self, z):
        # 处理z
        z_t = z[:, 0, :]  # [N, distil_size]
        z_p = z[:, 1, :]  # [N, disperse_size]
        if self.only_t:
            z_mem = self.z_memory_layer(z_t)  # [N, self.emb_size]
            z_mem = z_mem.unsqueeze(1)  # [N, 1, self.emb_size]
        else:
            assert z_t.shape[1] == z_p.shape[1], 'disperse size is not equal to distill size'
            z_mem = self.z_memory_layer(z)
        z_emb = self.z_embedding_layer(z_p)  # [N, self.hid_emb]
        return z_mem, z_emb

    def get_generate_idx(self, batch_size, device):
        # print('get index by generate')
        build_inputs = torch.zeros([batch_size, self.max_len-1])
        build_inputs[:, 1] = self.bos_idx
        build_inputs[:, 1:] = self.mask_idx
        build_inputs = build_inputs.to(device)
        return build_inputs

    def get_input_idx(self, inputs):
        # print('get index by idx')
        build_inputs = self.get_generate_idx(inputs.shape[0], inputs.device)
        return build_inputs

    def get_input(self, inputs, z):
        """
        获取input emb
        step1: 将input设置为mask的形式，长度为max_len - 1, 然后做嵌入, 然后把z_mem concat上去
        step2: 设置positional encoding
        step3: 把z_emb的形状阔转
        step4: concat上面三个向量 [max_len + z_len - 1, N, 3*E]
        output：通过一个linear层match dimension [max_len + z_len - 1, N, E]
        inputs:
            inputs: torch.Tensor [N, max_len]
            z: torch.Tensor [N, 2, E]
        outputs:
            embeds: torch.Tensor [max_len + z_len - 1, N, E]
            z_len: int
        """
        # 获取z_mem和z_emb
        # z_mem: [N, 1 or 2, E]
        # z_emb: [N, E]
        z_mem, z_emb = self.deal_z(z)
        # step 1 获取输入idx并做嵌入, z后面跟了max_len - 1个mask，预测除bos外第一个字到eos的所有词
        input_idx = self.get_input_idx(inputs)
        input_matrix = self.emb_model(input_idx.long()).to(z_mem.device)
        input_matrix = torch.cat([z_mem, input_matrix], dim=1).permute(1, 0, 2)
        # input_matrix: [seq_len + z_len, batch_size, hid_emb]
        # pos encoding
        pos_enc = torch.zeros(input_matrix.shape).to(z.device)
        pos_enc = self.pos_encoder(pos_enc)
        # z_emb
        z_emb = z_emb.unsqueeze(1)
        z_emb = z_emb.repeat(1, input_matrix.shape[0], 1).permute(1, 0, 2)  # [seq_len + z_len, batch_size, hid_emb]
        # concat final emb
        embeds = torch.cat([z_emb, input_matrix, pos_enc], dim=2)  # [seq_len + z_len, N, 3emb_size]
        embeds = self.small_hid_emb_layer(embeds)  # [seq_len + z_len, N, emb_size]
        return embeds, z_mem.shape[1]

    def tgt_to_output(self, tgt):
        # output = checkpoint(self.output_layer, tgt)
        output = self.output_layer(tgt)
        return output

    def one_decoder_step(self, inputs, z_len, attn_mask, key_padding_mask):
        """
        inputs: [seq_len, batch_size, hid_emb]
        attn_mask: attention mask (triangle mask)
        key_padding_mask: padding mask
        z_fix: whether fix z in the decoder layer
        """
        # output = checkpoint(self.gpt_decoder, inputs, attn_mask, key_padding_mask, z_fix)  # [S, N, hid_emb]
        x = inputs[z_len:, :, :]
        z = inputs[:z_len, :, :]
        # print(x.shape, z.shape, attn_mask.shape)
        output = self.gpt_decoder(x=x, z=z, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask, z_fix=self.z_fix)  # [S, N, hid_emb]
        output = output.permute(1, 0, 2)  # [N, S, hid_emb]
        return output

    def generate_one_sentence(self, inputs, z, tgt_key_padding_mask=None):
        '''
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 获取输入
        embeds, z_len = self.get_input(inputs, z)  # [seq_len + z_len, N, 3emb_size]
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0]).to(z.device)
        # output result: [N, seq_len + z_len, emb_size]
        output = self.one_decoder_step(inputs=embeds, z_len=z_len, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
        output = output[:, z_len:, :]  # [N, seq_len, emb_size]
        # get logits
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        # add bos
        bos_prob = self.get_bos_prob(output_prob.shape[0])
        output_prob = torch.cat([bos_prob.to(output_prob.device), output_prob], dim=1)
        return output_prob

    def generate(self, z):
        output_prob = self.generate_one_sentence(inputs=None, z=z, tgt_key_padding_mask=None)
        return output_prob

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            if self.only_t:
                z_padding_mask = torch.ones(pad_mask.shape[0], 1).to(pad_mask.device)
            else:
                z_padding_mask = torch.ones(pad_mask.shape[0], 2).to(pad_mask.device)
            pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
            tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        if is_training:
            # forward the GPT model
            output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output_prob = self.generate(z)
        return output_prob


class GPT_EASY(GPT):
    """
    the full GPT language model, with a context size of block_size, But train in easy mode
    decoder_new_embedding: Build a new embedding layer or use the trained one
    decoder_z_fix: fix z in the GPT training or not
    decoder_only_t: use only z_t as the head of the gpt or use z_t + z_p
    """

    def __init__(self, config, device, bos, emb_layer, decoder_new_embedding: bool = True, decoder_z_fix: bool = True,
                 decoder_only_t: bool = True, **kwargs):
        super().__init__(config, device, bos, emb_layer, decoder_new_embedding, decoder_z_fix, decoder_only_t, **kwargs)

    def get_input_idx(self, inputs):
        # print('get index by idx')
        build_inputs = inputs[:, :-1]
        return build_inputs

    def get_last_word(self, output_prob):
        '''
        output_prob: [Num, seq_len, num_tokens]
        '''
        next_prob = output_prob[:, -1].unsqueeze(1)  # [N, 1, num_tokens]
        output_idx = torch.argmax(output_prob, dim=2)[:, -1]  # [N, max_len-1]
        return next_prob, output_idx

    def generate_one_sentence(self, inputs, z, tgt_key_padding_mask=None):
        '''
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 获取输入
        embeds, z_len = self.get_input(inputs, z)  # [seq_len + z_len, N, 3emb_size]
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0]).to(z.device)
        # output result: [N, seq_len + z_len, emb_size]
        output = self.one_decoder_step(inputs=embeds, z_len=z_len, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
        output = output[:, z_len:, :]  # [N, seq_len, emb_size]
        # get logits
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        # add bos
        bos_prob = self.get_bos_prob(output_prob.shape[0])
        output_prob = torch.cat([bos_prob.to(output_prob.device), output_prob], dim=1)
        return output_prob

    def generate(self, z):
        # first word bos
        next_input = torch.ones(z.shape[0], 1).fill_(self.bos_idx).type(torch.long).to(z.device)  # [N, 1]
        eos_input = torch.ones(z.shape[0], 1).fill_(self.eos_idx).type(torch.long).to(z.device)  # [N, 1]
        # start bos prob
        bos_prob = self.get_bos_prob(z.shape[0]).to(z.device)
        total_sentence_probs = [bos_prob]
        for _ in range(self.max_len-1):
            # 从input生成, 由于会去掉最后一个word，所以加一个eos一起输进去
            input_idx = torch.cat([next_input, eos_input], dim=1)
            output_prob = self.generate_one_sentence(inputs=input_idx, z=z, tgt_key_padding_mask=None)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        """输入input idx与z，输出每个word的probability，不用match network

        Args:
            inputs (_type_): _description_
            z (_type_): _description_
            pad_mask (_type_, optional): _description_. Defaults to None.
            is_training (bool, optional): _description_. Defaults to True.
            easy (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            if self.only_t:
                z_padding_mask = torch.ones(pad_mask.shape[0], 1).to(pad_mask.device)
            else:
                z_padding_mask = torch.ones(pad_mask.shape[0], 2).to(pad_mask.device)
            pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
            tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        if is_training:
            # forward the GPT model
            output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output_prob = self.generate(z)
        return output_prob


class GPT_EASY_DVQ(GPT_EASY):
    """
    the full GPT language model, with a context size of block_size, But train in easy mode
    decoder_new_embedding: Build a new embedding layer or use the trained one
    decoder_z_fix: fix z in the GPT training or not
    decoder_only_t: use only z_t as the head of the gpt or use z_t + z_p
    """

    def __init__(self, config, device, bos, emb_layer, decoder_new_embedding: bool = True, decoder_z_fix: bool = False,
                 decoder_only_t: bool = False, **kwargs):
        super().__init__(config, device, bos, emb_layer, decoder_new_embedding, decoder_z_fix, decoder_only_t, **kwargs)

    def deal_z(self, z):
        """_summary_

        Args:
            z (torch.tensor 2d): [batch size, z_len, emb_size]

        Returns:
            _type_: _description_
        """
        # 处理z
        z_t = z[:, 0, :].unsqueeze(1)  # [N, 1, distil_size]
        z_p = z[:, 0:, :]  # [N, z_len-1, disperse_size]
        if self.only_t:
            z_mem = self.z_memory_layer(z_t)  # [N, 1, self.emb_size]
        else:
            z_mem = self.z_memory_layer(z)  # [N, z_len ,self.emb_size]
        z_emb = self.z_embedding_layer(z_p)  # [N, z_len-1 ,self.emb_size]
        return z_mem, z_emb

    def get_input(self, inputs, z):
        """
        获取input emb
        step1: get input idx
        step2: add positional encoding
        output：通过一个linear层match dimension [max_len + z_len - 1, N, E]
        inputs:
            inputs: torch.Tensor [N, max_len]
            z: torch.Tensor [N, 2, E]
        outputs:
            embeds: torch.Tensor [max_len + z_len - 1, N, E]
            z_len: int
        """
        # 获取z_mem和z_emb
        # z_mem: [N, z_len-1 or z_len, E]
        z_mem, _ = self.deal_z(z)
        # step 1 concat z mem and input
        input_idx = self.get_input_idx(inputs)
        input_matrix = self.emb_model(input_idx.long()).to(z_mem.device)
        # input_matrix: [seq_len + z_len -1 , batch_size, hid_emb]
        input_matrix = torch.cat([z_mem, input_matrix], dim=1).permute(1, 0, 2)
        # pos encoding
        input_matrix = self.pos_encoder(input_matrix)
        return input_matrix, z_mem.shape[1]

    def forward(self, inputs, z, pad_mask=None, is_training=True, **kwargs):
        """输入input idx与z，输出每个word的probability，不用match network

        Args:
            inputs (_type_): _description_
            z (_type_): _description_
            pad_mask (_type_, optional): _description_. Defaults to None.
            is_training (bool, optional): _description_. Defaults to True.
            easy (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            z_len = z.shape[1]
            if self.only_t:
                z_padding_mask = torch.ones(pad_mask.shape[0], z_len - 1).to(pad_mask.device)
            else:
                z_padding_mask = torch.ones(pad_mask.shape[0], z_len).to(pad_mask.device)
            pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
            tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        if is_training:
            # forward the GPT model
            output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output_prob = self.generate(z)
        return output_prob

class GPT_Emb_EASY(GPTModelBase):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = True, z_fix: bool = True,
                 only_t: bool = False, **kwargs):
        super().__init__(config, device, bos, emb_layer, new_embedding, z_fix, only_t, **kwargs)
        # vocab setting
        self.emb_layer = emb_layer
        self.bos = self.emb_layer.emb_model.get_bos()

        if self.only_t:
            self.pos_encoder = PositionalEncoding(self.hid_emb, self.dropout_rate, self.max_len)
        else:
            self.pos_encoder = PositionalEncoding(self.hid_emb, self.dropout_rate, self.max_len+1)
        self.drop = nn.Dropout(self.dropout_rate)
        # GPT
        gpt_layer = GPTLayer(self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hid_emb, self.emb_size, bias=False)
        # z memory and emb layer
        self.z_embedding_layer = nn.Linear(config.distil_size, self.hid_emb)
        self.z_memory_layer = nn.Linear(config.disper_size, self.hid_emb)
        # emb_layer
        concat_emb = 3 * self.hid_emb
        self.small_hid_emb_layer = nn.Linear(concat_emb, self.hid_emb)

        self._reset_parameters()

    def deal_z(self, z):
        # 处理z
        z_t = z[:, 0, :]  # [N, distil_size]
        z_p = z[:, 1, :]  # [N, disperse_size]
        if self.only_t:
            z_mem = self.z_memory_layer(z_t)  # [N, self.emb_size]
            z_mem = z_mem.unsqueeze(1)  # [N, 1, self.emb_size]
        else:
            assert z_t.shape[1] == z_p.shape[1], 'disperse size is not equal to distill size'
            z_mem = self.z_memory_layer(z)
        z_emb = self.z_embedding_layer(z_p)  # [N, self.hid_emb]
        return z_mem, z_emb

    def get_input_idx(self, inputs):
        # print('get index by idx')
        build_inputs = inputs[:, :-1, :]
        return build_inputs

    def get_input(self, inputs, z):
        """
        获取input emb
        step1: z_mem和input concat
        step2: 设置positional encoding
        step3: 把z_emb的形状阔转
        step4: concat上面三个向量 [max_len + z_len - 1, N, 3*E]
        output：通过一个linear层match dimension [max_len + z_len - 1, N, E]
        inputs:
            inputs: torch.Tensor [N, max_len, emb_size]
            z: torch.Tensor [N, 2, E]
        outputs:
            embeds: torch.Tensor [max_len + z_len - 1, N, E]
            z_len: int
        """
        # 获取z_mem和z_emb
        # z_mem: [N, 1 or 2, E]
        # z_emb: [N, E]
        z_mem, z_emb = self.deal_z(z)
        # step 1 concat z mem and input
        inputs = self.get_input_idx(inputs)
        input_matrix = torch.cat([z_mem, inputs], dim=1).permute(1, 0, 2)
        # input_matrix: [seq_len + z_len -1 , batch_size, hid_emb]
        # pos encoding
        pos_enc = torch.zeros(input_matrix.shape).to(z.device)
        pos_enc = self.pos_encoder(pos_enc)
        # z_emb
        z_emb = z_emb.unsqueeze(1)
        z_emb = z_emb.repeat(1, input_matrix.shape[0], 1).permute(1, 0, 2)  # [seq_len + z_len - 1, batch_size, hid_emb]
        # concat final emb
        embeds = torch.cat([z_emb, input_matrix, pos_enc], dim=2)  # [seq_len + z_len - 1, N, 3emb_size]
        embeds = self.small_hid_emb_layer(embeds)  # [seq_len + z_len - 1, N, emb_size]
        return embeds, z_mem.shape[1]

    def tgt_to_output(self, tgt):
        output = self.output_layer(tgt)
        return output

    def one_decoder_step(self, inputs, z_len, attn_mask, key_padding_mask):
        """
        inputs: [seq_len, batch_size, hid_emb]
        attn_mask: attention mask (triangle mask)
        key_padding_mask: padding mask
        z_fix: whether fix z in the decoder layer
        """
        # output = checkpoint(self.gpt_decoder, inputs, attn_mask, key_padding_mask, z_fix)  # [S, N, hid_emb]
        x = inputs[z_len:, :, :]
        z = inputs[:z_len, :, :]
        # print(x.shape, z.shape, attn_mask.shape)
        output = self.gpt_decoder(x=x, z=z, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask, z_fix=self.z_fix)  # [S, N, hid_emb]
        output = output.permute(1, 0, 2)  # [N, S, hid_emb]
        return output

    def generate_one_sentence(self, inputs, z, tgt_key_padding_mask=None):
        '''
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 获取输入
        embeds, z_len = self.get_input(inputs, z)  # [seq_len + z_len, N, 3emb_size]
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0]).to(z.device)
        # output result: [N, seq_len + z_len, emb_size]
        output = self.one_decoder_step(inputs=embeds, z_len=z_len, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
        output = output[:, z_len:, :]  # [N, seq_len, emb_size]
        # get logits
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        # add bos
        bos_matrix = self.bos.repeat(output.shape[0], 1, 1).to(output.device)  # [N, 1, 1]
        output_prob = torch.cat([bos_matrix, output_prob], dim=1)
        return output_prob

    def generate(self, z):
        # add bos
        bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, 1]
        next_input = bos_matrix
        for _ in range(self.max_len-1):
            # 从input生成, 由于会去掉最后一个word，所以加一个长度为1的一起输进去
            inputs = torch.cat([next_input, bos_matrix], dim=1)
            output = self.generate_one_sentence(inputs=inputs, z=z, tgt_key_padding_mask=None)
            # 获取最后一个词的emb
            output_embs = output[:, -1, :].unsqueeze(1)
            # emb放入
            next_input = torch.cat([next_input, output_embs], dim=1)  # [N, now_len+1, emb_size]
        return next_input

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        """输入embedding和z，输出embedding而不是probability，没有match network

        Args:
            inputs (_type_): _description_
            z (_type_): _description_
            pad_mask (_type_, optional): _description_. Defaults to None.
            is_training (bool, optional): _description_. Defaults to True.
            easy (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            # 由于有z在前面，所以需要在padding前面加上1/2个
            if self.only_t:
                z_padding_mask = torch.ones(pad_mask.shape[0], 1).to(pad_mask.device)
            else:
                z_padding_mask = torch.ones(pad_mask.shape[0], 2).to(pad_mask.device)
            pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
            tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        if is_training:
            # forward the GPT model
            output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output_prob = self.generate(z)
        return output_prob


class GPT_Emb_Match_Network_EASY(GPT):
    """
    GPT Match Network版本，使用词向量先验，gpt模型输出一个向量，经过一个match network获取probability
    """

    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = False, z_fix: bool = True,
                 only_t: bool = False, **kwargs):

        super().__init__(config, device, bos, emb_layer, new_embedding, z_fix, only_t)
        # GPT
        gpt_layer = GPTLayer(2 * self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)

        self.output_layer = nn.Linear(2 * self.hid_emb, self.emb_size, bias=False)
        self.emb_weight = self.emb_layer.emb_model.embedding_layer.weight

    def match_network(self, output):
        """
        利用output和embedding计算cos similarity来获得index
        Inputs:
            output: [B, S, emb_size]
        Outputs:
            prob: [B, S, NTokens]
        """
        sim_output_list = []
        for i in range(output.shape[0]):
            sim_vec = output[i, :, :].unsqueeze(1)
            sim = F.cosine_similarity(sim_vec,  # [S, 1, emb_size]
                                      self.emb_weight.unsqueeze(0).to(sim_vec.device),  # [1, NTokens, emb_size]
                                      dim=2)
            sim = sim.reshape(1, output.shape[1], -1)  # [1, S, NTokens]
            sim_output_list.append(sim)
        sim_output = torch.cat(sim_output_list, dim=0)
        return sim_output

    def match_network_ed(self, output):
        sim_output_list = []
        emb_weight = self.emb_weight.to(output.device).detach()
        for i in range(output.shape[0]):
            sim_vec = output[i, :, :].unsqueeze(1)
            sim = torch.cdist(sim_vec, emb_weight)
            sim = sim.reshape(1, output.shape[1], -1)  # [1, S, NTokens]
            sim_output_list.append(sim)
        sim_output = torch.cat(sim_output_list, dim=0)
        # sim_output = torch.cdist(output, self.emb_weight)
        return sim_output

    def tgt_to_output(self, tgt):
        tgt = self.output_layer(tgt)
        # output_prob = self.match_network(tgt)
        output_prob = self.match_network_ed(tgt)
        return output_prob

    def get_last_word(self, output_prob):
        '''
        output_prob: [Num, seq_len, num_tokens]
        '''
        next_prob = output_prob[:, -1, :].unsqueeze(1)  # [N, 1, num_tokens]
        output_idx = torch.argmax(output_prob, dim=2)[:, -1]  # [N, max_len-1]
        return next_prob, output_idx

    def get_input_idx(self, inputs):
        # print('get index by idx')
        build_inputs = inputs[:, :-1]
        return build_inputs

    def merge_input_matrix(self, input_matrix, z_emb, pos_enc):
        input_matrix = input_matrix + z_emb
        return torch.cat([input_matrix, pos_enc], dim=2)  # [seq_len + z_len - 1, N, 3emb_size]

    def get_input(self, inputs, z):
        """
        获取input emb
        step1: z_mem和input concat
        step2: 设置positional encoding
        step3: 把z_emb的形状阔转
        step4: concat上面三个向量 [max_len + z_len - 1, N, 3*E]
        output：通过一个linear层match dimension [max_len + z_len - 1, N, E]
        inputs:
            inputs: torch.Tensor [N, max_len, emb_size]
            z: torch.Tensor [N, 2, E]
        outputs:
            embeds: torch.Tensor [max_len + z_len - 1, N, E]
            z_len: int
        """
        # 获取z_mem和z_emb
        # z_mem: [N, 1 or 2, E]
        # z_emb: [N, E]
        z_mem, z_emb = self.deal_z(z)
        # step 1 concat z mem and input
        input_idx = self.get_input_idx(inputs)
        input_matrix = self.emb_model(input_idx.long()).to(z_mem.device)
        input_matrix = torch.cat([z_mem, input_matrix], dim=1).permute(1, 0, 2)
        # input_matrix: [seq_len + z_len -1 , batch_size, hid_emb]
        # pos encoding
        pos_enc = torch.zeros(input_matrix.shape).to(z.device)
        pos_enc = self.pos_encoder(pos_enc)
        # z_emb
        z_emb = z_emb.unsqueeze(1)
        z_emb = z_emb.repeat(1, input_matrix.shape[0], 1).permute(1, 0, 2)  # [seq_len + z_len - 1, batch_size, hid_emb]
        # concat final emb
        embeds = self.merge_input_matrix(input_matrix, z_emb, pos_enc)
        return embeds, z_mem.shape[1]

    def generate_one_sentence(self, inputs, z, tgt_key_padding_mask=None):
        '''
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 获取输入
        embeds, z_len = self.get_input(inputs, z)  # [seq_len + z_len, N, 3emb_size]
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0]).to(z.device)
        # output result: [N, seq_len + z_len, emb_size]
        output = self.one_decoder_step(inputs=embeds, z_len=z_len, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
        output = output[:, z_len:, :]  # [N, seq_len, emb_size]
        # get logits
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        # add bos
        bos_prob = self.get_bos_prob(output_prob.shape[0])
        output_prob = torch.cat([bos_prob.to(output_prob.device), output_prob], dim=1)
        return output_prob

    def generate(self, z):
        # first word bos
        next_input = torch.ones(z.shape[0], 1).fill_(self.bos_idx).type(torch.long).to(z.device)  # [N, 1]
        eos_input = torch.ones(z.shape[0], 1).fill_(self.eos_idx).type(torch.long).to(z.device)  # [N, 1]
        # start bos prob
        bos_prob = self.get_bos_prob(z.shape[0]).to(z.device)
        total_sentence_probs = [bos_prob]
        for _ in range(self.max_len-1):
            # 从input生成, 由于会去掉最后一个word，所以加一个eos一起输进去
            input_idx = torch.cat([next_input, eos_input], dim=1)
            output_prob = self.generate_one_sentence(inputs=input_idx, z=z, tgt_key_padding_mask=None)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True, pretrain=False):
        if pretrain:
            output_prob = self.pretrain(inputs, pad_mask)
        else:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                # add z numbers
                if self.only_t:
                    z_padding_mask = torch.ones(pad_mask.shape[0], z.shape[1] - 1).to(pad_mask.device)
                else:
                    z_padding_mask = torch.ones(pad_mask.shape[0], z.shape[1]).to(pad_mask.device)
                pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
            if is_training and self.training:
                # forward the GPT model
                output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
            else:
                output_prob = self.generate(z)
        return output_prob

    def pretrain(self, inputs, tgt_key_padding_mask=None):
        """
        pretrain decoder, 不用z，只训练decoder上下文, z_emb用0代替
        """
        # step 1 concat z mem and input
        input_idx = self.get_input_idx(inputs)
        input_matrix = self.emb_model(input_idx.long()).to(inputs.device).permute(1, 0, 2)
        # input_matrix: [seq_len + z_len -1 , batch_size, hid_emb]
        # pos encoding
        pos_enc = torch.zeros(input_matrix.shape).to(inputs.device)
        pos_enc = self.pos_encoder(pos_enc)
        # z_emb
        z_emb = torch.zeros(input_matrix.shape).to(inputs.device)
        # concat final emb
        embeds = self.merge_input_matrix(input_matrix, z_emb, pos_enc)
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0]).to(inputs.device)
        # output result: [N, seq_len + z_len, emb_size]
        tgt_key_padding_mask = (tgt_key_padding_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        output = self.gpt_decoder.pretrain(x=embeds, attn_mask=tgt_mask,
                                           key_padding_mask=tgt_key_padding_mask)  # [S, N, hid_emb]
        output = output.permute(1, 0, 2)  # [N, S, hid_emb]
        # get logits
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        # add bos
        bos_prob = self.get_bos_prob(output_prob.shape[0])
        output_prob = torch.cat([bos_prob.to(output_prob.device), output_prob], dim=1)
        return output_prob


class GPT_Emb_Match_Network_EASY_Concat2(GPT_Emb_Match_Network_EASY):
    """
    GPT Match Network版本，concat [input matrix, pos_enc]
    
    """

    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = False, z_fix: bool = False,
                 only_t: bool = False, **kwargs):

        super().__init__(config, device, bos, emb_layer, new_embedding, z_fix, only_t)
        # GPT
        gpt_layer = GPTLayer(2 * self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)

        self.output_layer = nn.Linear(2 * self.hid_emb, self.emb_size, bias=False)
        self.emb_weight = self.emb_layer.emb_model.embedding_layer.weight

    def merge_input_matrix(self, input_matrix, z_emb, pos_enc):
        return torch.cat([input_matrix, pos_enc], dim=2)  # [seq_len + z_len - 1, N, 3emb_size]


class GPT_Emb_Match_Network_EASY_Concat3(GPT_Emb_Match_Network_EASY):
    """
    GPT Match Network版本，concat [input matrix, pos_enc, z_emb]
    
    """

    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = False, z_fix: bool = False,
                 only_t: bool = False, **kwargs):

        super().__init__(config, device, bos, emb_layer, new_embedding, z_fix, only_t)
        # GPT
        gpt_layer = GPTLayer(3 * self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)

        self.output_layer = nn.Linear(3 * self.hid_emb, self.emb_size, bias=False)
        self.emb_weight = self.emb_layer.emb_model.embedding_layer.weight

    def merge_input_matrix(self, input_matrix, z_emb, pos_enc):
        return torch.cat([input_matrix, pos_enc, z_emb], dim=2)  # [seq_len + z_len - 1, N, 3emb_size]
    

class GPT_Emb_Match_Network_DVQ(GPT_Emb_Match_Network_EASY):
    """
    GPT Match Network，DVQ版本，输入的z的维度高于1维
    """

    def __init__(self, config, device, bos, emb_layer, new_embedding: bool = False, z_fix: bool = False,
                 only_t: bool = False, **kwargs):

        super().__init__(config, device, bos, emb_layer, new_embedding, z_fix, only_t)
        # GPT
        gpt_layer = GPTLayer(self.hid_emb, self.num_heads, self.dropout_rate, self.forward_expansion)
        self.gpt_decoder = GPTDecoder(decoder_layer=gpt_layer, num_layers=self.num_layers)

        self.output_layer = nn.Linear(self.hid_emb, self.emb_size, bias=False)
        self.emb_weight = self.emb_layer.emb_model.embedding_layer.weight

    def deal_z(self, z):
        """_summary_

        Args:
            z (torch.tensor 2d): [batch size, z_len, emb_size]

        Returns:
            _type_: _description_
        """
        # 处理z
        z_t = z[:, 0, :].unsqueeze(1)  # [N, 1, distil_size]
        z_p = z[:, 0:, :]  # [N, z_len-1, disperse_size]
        if self.only_t:
            z_mem = self.z_memory_layer(z_t)  # [N, 1, self.emb_size]
        else:
            z_mem = self.z_memory_layer(z)  # [N, z_len ,self.emb_size]
        z_emb = self.z_embedding_layer(z_p)  # [N, z_len-1 ,self.emb_size]
        return z_mem, z_emb

    def get_input(self, inputs, z):
        """
        获取input emb
        step1: get input idx
        step2: add positional encoding
        output：通过一个linear层match dimension [max_len + z_len - 1, N, E]
        inputs:
            inputs: torch.Tensor [N, max_len]
            z: torch.Tensor [N, 2, E]
        outputs:
            embeds: torch.Tensor [max_len + z_len - 1, N, E]
            z_len: int
        """
        # 获取z_mem和z_emb
        # z_mem: [N, z_len-1 or z_len, E]
        z_mem, _ = self.deal_z(z)
        # step 1 concat z mem and input
        input_idx = self.get_input_idx(inputs)
        input_matrix = self.emb_model(input_idx.long()).to(z_mem.device)
        # input_matrix: [seq_len + z_len -1 , batch_size, hid_emb]
        input_matrix = torch.cat([z_mem, input_matrix], dim=1).permute(1, 0, 2)
        # pos encoding
        input_matrix = self.pos_encoder(input_matrix)
        return input_matrix, z_mem.shape[1]

    def forward(self, inputs, z, pad_mask=None, is_training=True, **kwargs):
        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            # add z numbers
            z_len = z.shape[1]
            if self.only_t:
                z_padding_mask = torch.ones(pad_mask.shape[0], z_len - 1).to(pad_mask.device)
            else:
                z_padding_mask = torch.ones(pad_mask.shape[0], z_len).to(pad_mask.device)
            pad_mask = torch.cat([z_padding_mask, pad_mask], dim=1)
            tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        if is_training and self.training:
            # forward the GPT model
            output_prob = self.generate_one_sentence(inputs, z, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output_prob = self.generate(z)
        return output_prob