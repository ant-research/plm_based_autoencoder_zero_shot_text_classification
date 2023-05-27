import torch
import torch.nn as nn
from model.new_model.modules.simple_modules import PositionalEncoding
from torch.utils.checkpoint import checkpoint


class Transformer(nn.Module):
    def __init__(self, config, device, decoder_name='transformerdecoder', bos=None):
        '''
        input_size: the distiling + dispersing size
        '''
        super(Transformer, self).__init__()
        if config.emb_size % config.num_heads != 0:
            raise ValueError("Embedding size %d needs to be divisible by number of heads %d"
                             % (config.emb_size, config.num_heads))
        self.emb_size = config.emb_size  # embedding size of word embedding
        self.device = device
        self.k = config.class_num
        # self.hid_emb = 256
        self.hid_emb = self.emb_size
        self.max_len = config.seq_len
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        self.z_size = config.distil_size + config.disper_size
        self.decoder_name = decoder_name
        # beginning of the sentence vector
        if bos is None:
            self.bos = torch.zeros([1, 1, self.emb_size])  # [1, 1, emb_size]
        else:
            self.bos = bos

        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))
        
        self.pos_encoder = PositionalEncoding(self.emb_size, self.dropout_rate, self.max_len)
        # Transformer dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        self.z_layer = nn.Linear(self.z_size, self.hid_emb)
        # self.input_layer = nn.Linear(self.emb_size, self.hid_emb, bias=True)
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                   dim_feedforward=self.forward_expansion*self.hid_emb,
                                                   dropout=self.dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        # self.output_layer = nn.Linear(self.hid_emb, self.emb_size)
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        # Generate mask covering the top right triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def input_to_tgt(self, inputs):
        # tgt = self.input_layer(inputs)
        tgt = inputs
        return tgt

    def tgt_to_output(self, tgt):
        # output = checkpoint(self.output_layer, tgt)
        output = tgt
        return output

    def one_decoder_step(self, z, inputs, tgt_mask=None, tgt_key_padding_mask=None):
        # memory = z.repeat(inputs.shape[0], 1, 1) # 扩充至句子长度 [S, N, hid_emb]
        memory = z
        tgt = self.input_to_tgt(inputs)  # 维度映射层 [S, N, hid_emb]
        output = checkpoint(self.decoder, tgt, memory, tgt_mask, None, tgt_key_padding_mask)  # [S, N, hid_emb]
        output = output.permute(1, 0, 2)  # [N, S, hid_emb]
        output = self.tgt_to_output(output)  # [N, max_len-1, E]
        return output

    def forward(self, x, z, pad_mask=None, is_training=True, easy=True):
        '''
        inputs: [N, S, E]
        z: [N, dist_size + dispersing]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if is_training and self.training:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            tgt_mask = self.generate_square_subsequent_mask(x.shape[1]-1)
            tgt_mask = tgt_mask.to(z.device)

            # 处理z
            z = self.z_layer(z)  # [N, hid_emb]
            z = z.unsqueeze(0)  # [1, N, hid_emb]

            # add positional encoding
            x = x[:, :-1, :]
            x = x.permute(1, 0, 2)  # [S, N, E]
            # embeds = self.pos_encoder(x * math.sqrt(self.emb_size))  # [S, N, E]
            embeds = self.pos_encoder(x)
            embeds = self.dropout(embeds)

            output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)

            bos_matrix = self.bos.repeat(output.shape[0], 1, 1).to(output.device)  # [N, 1, 1]
            output = torch.cat([bos_matrix, output], dim=1)   # [N, max_len, E]
        else:
            bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, E]
            z = self.z_layer(z)  # [N, hid_emb]
            z = z.unsqueeze(0)  # [1, N, hid_emb]
            next_input = bos_matrix  # [N, 1, E]
            for _ in range(self.max_len-1):
                inputs = next_input.permute(1, 0, 2).detach()
                # embeds = checkpoint(self.pos_encoder, inputs * math.sqrt(self.emb_size))  # [now_len, N, hid_emb]
                embeds = checkpoint(self.pos_encoder, inputs)  # [now_len, N, hid_emb]
                # get mask
                tgt_mask = self.generate_square_subsequent_mask(inputs.shape[0])
                tgt_mask = tgt_mask.to(z.device)
                output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=None)
                next_input = torch.cat([next_input, output[:, -1, :].unsqueeze(1)], dim=1)  # [N, now_len+1, emb_size]
            output = next_input
        return output


class TransformerIdx(Transformer):
    def __init__(self, config, device, decoder_name='transformer_idx decoder', bos=None, emb_layer=None):
        '''
        input_size: the distiling + dispersing size
        '''
        super(TransformerIdx, self).__init__(config, device, decoder_name=decoder_name, bos=bos)
        self.emb_layer = emb_layer
        self.token_num = len(self.emb_layer.emb_model.index_vocab_dict)
        self.bos_idx = self.emb_layer.emb_model.bos_idx
        self.eos_idx = self.emb_layer.emb_model.eos_idx

        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))

        self.pos_encoder = PositionalEncoding(self.emb_size, self.dropout_rate, self.max_len)
        # Transformer dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # self.z_layer = nn.Linear(self.z_size, self.hid_emb)
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                   dim_feedforward=self.forward_expansion*self.hid_emb,
                                                   dropout=self.dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.output_layer = nn.Linear(self.hid_emb, self.token_num)
        self._init_weights()

    def get_bos_prob(self, shape):
        bos_prob = torch.zeros([shape, 1, self.token_num])
        bos_prob[:, :, self.bos_idx] = 1
        return bos_prob

    def input_to_tgt(self, inputs):
        # tgt = self.input_layer(inputs)
        tgt = inputs
        return tgt

    def tgt_to_output(self, tgt):
        output = checkpoint(self.output_layer, tgt)
        #output = tgt
        return output

    def get_last_word(self, output_prob):
        '''
        output_prob: [Num, seq_len, num_tokens]
        '''
        next_prob = output_prob[:, -1].unsqueeze(1)  # [N, 1, num_tokens]
        output_idx = torch.argmax(output_prob, dim=2)[:, -1]  # [N, max_len-1]
        return next_prob, output_idx

    def one_decoder_step(self, z, inputs, tgt_mask=None, tgt_key_padding_mask=None):
        # memory = z.repeat(inputs.shape[0], 1, 1)  # 扩充至句子长度 [S, N, hid_emb]
        memory = z
        tgt = self.input_to_tgt(inputs)  # 维度映射层 [S, N, hid_emb]
        # output = checkpoint(self.decoder, tgt, memory, tgt_mask, None, tgt_key_padding_mask)  # [S, N, hid_emb]
        output = self.decoder(tgt, memory, tgt_mask, None, tgt_key_padding_mask)  # [S, N, hid_emb]
        output = output.permute(1, 0, 2)  # [N, S, hid_emb]
        return output

    def deal_z(self, z):
        # 处理z
        z = z.permute(1, 0, 2)  # [2, N, hid_emb]
        return z

    def generate_one_sentence(self, input_idx, z, tgt_key_padding_mask=None):
        '''
        input_idx: [N, max_len-1]
        '''
        # 从inputs idx中输出embedding
        embeds = self.emb_layer.emb_model.emb_from_idx(input_idx).to(z.device)
        embeds = embeds.permute(1, 0, 2)
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0])
        tgt_mask = tgt_mask.to(z.device)
        # positional encoding
        # embeds = checkpoint(self.pos_encoder, embeds * math.sqrt(self.emb_size))  # [now_len, N, hid_emb]
        embeds = self.pos_encoder(embeds)  # [now_len, N, hid_emb]
        # output result
        output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        # add bos
        bos_matrix = self.bos.repeat(output.shape[0], 1, 1).to(output.device)  # [N, 1, 1]
        output = torch.cat([bos_matrix, output], dim=1)   # [N, max_len, E]
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        return output_prob

    def generate(self, z):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        # first word bos
        next_input = torch.ones(z.shape[0], 1).fill_(self.bos_idx).type(torch.long).to(z.device)  # [N, 1]
        # start bos prob
        bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, emb]
        bos_output_prob = self.tgt_to_output(bos_matrix)  # [N, max_len-1, num_tokens]
        bos_output_prob, output_idx = self.get_last_word(bos_output_prob)
        total_sentence_probs = [bos_output_prob]
        z = self.deal_z(z)
        for _ in range(self.max_len-1):
            # 从input生成
            output_prob = self.generate_one_sentence(next_input, z)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return next_input, total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        '''
        inputs: [N, S]
        z: [N, dist_size + dispersing]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if is_training and self.training:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            if easy is True:
                # if easy mode
                z = self.deal_z(z)
                inputs = inputs[:, :-1]
                output_prob = self.generate_one_sentence(input_idx=inputs, z=z,
                                                         tgt_key_padding_mask=tgt_key_padding_mask)
            else:
                # if difficult mode
                _, output_prob = self.generate(z)
            output = output_prob
        else:
            output, _ = self.generate(z)
        return output


class TransformerIdx_Mem(TransformerIdx):
    def __init__(self, config, device, decoder_name='transformer_idx_mem decoder', bos=None, emb_layer=None):
        '''
        input_size: the distiling + dispersing size
        add z to memory
        '''
        super(TransformerIdx_Mem, self).__init__(config, device, decoder_name='transformer_idx decoder', bos=bos, emb_layer=emb_layer)

        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))
        # OPTIMUS memory 向量
        self.z_embedding_layer = nn.Linear(self.z_size, self.emb_size)
        # OPTIMUS hidden state 向量
        self.z_memory_layer = nn.Linear(self.z_size, self.hid_emb)

    def deal_z(self, z):
        # 处理z
        z_t = z[:, 0, :]  # [N, z_len]
        z_p = z[:, 1, :]  # [N, z_len]
        z = torch.cat([z_t, z_p], dim=1)
        z_mem = self.z_memory_layer(z)  # [N, self.emb_size]
        z_emb = self.z_embedding_layer(z)  # [N, self.hid_emb]
        return z_mem, z_emb

    def generate_one_sentence(self, input_idx, z, z_emb, tgt_key_padding_mask=None):
        '''
        input_idx: [N, max_len-1]
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 从inputs idx中输出embedding
        embeds = self.emb_layer.emb_model.emb_from_idx(input_idx).to(z.device)  # [N, seq_len, emb_size]
        embeds = embeds + z_emb.unsqueeze(1)
        embeds = embeds.permute(1, 0, 2)
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0])
        tgt_mask = tgt_mask.to(z.device)
        # positional encoding
        # embeds = checkpoint(self.pos_encoder, embeds * math.sqrt(self.emb_size))  # [now_len, N, hid_emb]
        embeds = checkpoint(self.pos_encoder, embeds)  # [now_len, N, hid_emb]
        # output result
        output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        # add bos
        bos_matrix = self.bos.repeat(output.shape[0], 1, 1).to(output.device)  # [N, 1, 1]
        output = torch.cat([bos_matrix, output], dim=1)   # [N, max_len, E]
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        return output_prob

    def generate(self, z):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        # first word bos
        next_input = torch.ones(z.shape[0], 1).fill_(self.bos_idx).type(torch.long).to(z.device)  # [N, 1]
        # start bos prob
        bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, emb]
        bos_output_prob = self.tgt_to_output(bos_matrix)  # [N, max_len-1, num_tokens]
        bos_output_prob, output_idx = self.get_last_word(bos_output_prob)
        total_sentence_probs = [bos_output_prob]
        z_mem, z_emb = self.deal_z(z)
        for _ in range(self.max_len-1):
            # 从input生成
            output_prob = self.generate_one_sentence(next_input, z=z_mem, z_emb=z_emb)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return next_input, total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        '''
        inputs: [N, S]
        z: [N, dist_size + dispersing]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if is_training and self.training:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            if easy is True:
                # if easy mode
                z_mem, z_emb = self.deal_z(z)
                inputs = inputs[:, :-1]
                output_prob = self.generate_one_sentence(input_idx=inputs, z=z_mem, z_emb=z_emb,
                                                         tgt_key_padding_mask=tgt_key_padding_mask)
            else:
                # if difficult mode
                _, output_prob = self.generate(z)
            output = output_prob
        else:
            output, _ = self.generate(z)
        return output


class TransformerIdx_Mem_V2(TransformerIdx):
    def __init__(self, config, device, decoder_name='transformer_idx_mem decoder', bos=None, emb_layer=None):
        '''
        input_size: the distiling + dispersing size
        add z to memory
        '''
        super(TransformerIdx_Mem_V2, self).__init__(config, device, decoder_name='transformer_idx decoder', bos=bos, emb_layer=emb_layer)

        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))
        # OPTIMUS memory 向量
        self.z_embedding_layer = nn.Linear(config.distil_size, self.emb_size)
        # OPTIMUS hidden state 向量
        self.z_memory_layer = nn.Linear(config.disper_size, self.hid_emb)

    def deal_z(self, z):
        # 处理z
        z_t = z[:, 0, :]  # [N, z_len]
        z_p = z[:, 1, :]  # [N, z_len]
        z_mem = self.z_memory_layer(z_t)  # [N, self.emb_size]
        z_emb = self.z_embedding_layer(z_p)  # [N, self.hid_emb]
        return z_mem, z_emb

    def generate_one_sentence(self, input_idx, z, z_emb, tgt_key_padding_mask=None):
        '''
        input_idx: [N, max_len-1]
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 从inputs idx中输出embedding
        embeds = self.emb_layer.emb_model.emb_from_idx(input_idx).to(z.device)  # [N, seq_len, emb_size]
        embeds = embeds + z_emb.unsqueeze(1)
        embeds = embeds.permute(1, 0, 2)
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0])
        tgt_mask = tgt_mask.to(z.device)
        # positional encoding
        # embeds = checkpoint(self.pos_encoder, embeds * math.sqrt(self.emb_size))  # [now_len, N, hid_emb]
        embeds = checkpoint(self.pos_encoder, embeds)  # [now_len, N, hid_emb]
        # output result
        output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        # add bos
        bos_matrix = self.bos.repeat(output.shape[0], 1, 1).to(output.device)  # [N, 1, 1]
        output = torch.cat([bos_matrix, output], dim=1)   # [N, max_len, E]
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        return output_prob

    def generate(self, z):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        # first word bos
        next_input = torch.ones(z.shape[0], 1).fill_(self.bos_idx).type(torch.long).to(z.device)  # [N, 1]
        # start bos prob
        bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, emb]
        bos_output_prob = self.tgt_to_output(bos_matrix)  # [N, max_len-1, num_tokens]
        bos_output_prob, output_idx = self.get_last_word(bos_output_prob)
        total_sentence_probs = [bos_output_prob]
        z_mem, z_emb = self.deal_z(z)
        for _ in range(self.max_len-1):
            # 从input生成
            output_prob = self.generate_one_sentence(next_input, z=z_mem, z_emb=z_emb)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return next_input, total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        '''
        inputs: [N, S]
        z: [N, dist_size + dispersing]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if is_training and self.training:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            if easy is True:
                # if easy mode
                z_mem, z_emb = self.deal_z(z)
                inputs = inputs[:, :-1]
                output_prob = self.generate_one_sentence(input_idx=inputs, z=z_mem, z_emb=z_emb,
                                                         tgt_key_padding_mask=tgt_key_padding_mask)
            else:
                # if difficult mode
                _, output_prob = self.generate(z)
            output = output_prob
        else:
            output, _ = self.generate(z)
        return output


class InformerIdx(TransformerIdx):
    def __init__(self, config, device, decoder_name='informer_idx_mem decoder', bos=None, emb_layer=None):
        '''
        input_size: the distiling + dispersing size
        add z to memory
        '''
        super().__init__(config, device, decoder_name='transformer_idx decoder', bos=bos, emb_layer=emb_layer)
        print('Transformer setting: emb size %d, max len %d, class_num %d' % (self.emb_size, self.max_len, self.k))
        
        self.mask = torch.ones([1, 1, self.emb_size])  # [1, 1, emb_size]
        self.bos = self.emb_layer.get_bos()
        print(self.bos.shape)
        # OPTIMUS memory 向量
        self.z_embedding_layer = nn.Linear(config.distil_size, self.emb_size)
        # concat维度
        self.hid_emb = 3*self.hid_emb
        # OPTIMUS hidden state 向量
        self.z_memory_layer = nn.Linear(config.disper_size, self.hid_emb)
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_emb, nhead=self.num_heads,
                                                   dim_feedforward=self.forward_expansion*self.hid_emb,
                                                   dropout=self.dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.output_layer = nn.Linear(self.hid_emb, self.token_num)

    def deal_z(self, z):
        # 处理z
        z_t = z[:, 0, :]  # [N, z_len]
        z_p = z[:, 1, :]  # [N, z_len]
        z_mem = self.z_memory_layer(z_t)  # [N, self.emb_size]
        z_emb = self.z_embedding_layer(z_p)  # [N, self.hid_emb]
        return z_mem, z_emb

    def generate_one_sentence(self, input_idx, z, z_emb, tgt_key_padding_mask=None):
        '''
        input_idx: [N, max_len-1]
        z: z_mem [N, hid_emb]
        z_emb: [N, emb_size]
        '''
        # 获取输入
        bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # [N, 1, emb_size]
        mask_matrix = self.mask.repeat(z.shape[0], self.max_len-2, 1).to(z.device)  # [N, max_len-1, emb_size]
        input_matrix = torch.cat([bos_matrix, mask_matrix], dim=1).permute(1, 0, 2)  # [max_len, N, emb_size]
        # 获取 pos encoding
        pos_enc = torch.zeros(input_matrix.shape).to(z.device)
        pos_enc = self.pos_encoder(pos_enc)
        # z_emb
        z_emb = z_emb.unsqueeze(1)
        z_emb = z_emb.repeat(1, self.max_len-1, 1).permute(1, 0, 2)  # [max_len, N, emb_size]
        print(z_emb.shape, input_matrix.shape, pos_enc.shape)
        # 从inputs idx中输出embedding
        embeds = torch.cat([z_emb, input_matrix, pos_enc], dim=2)  # [max_len, N, 3emb_size]
        # get mask
        tgt_mask = self.generate_square_subsequent_mask(embeds.shape[0])
        tgt_mask = tgt_mask.to(z.device)
        # output result
        output = self.one_decoder_step(z, inputs=embeds, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        # add bos
        output_prob = self.tgt_to_output(output)  # [N, max_len-1, num_tokens]
        bos_prob = self.get_bos_prob(output.shape[0])
        output_prob = torch.cat([bos_prob.to(output_prob.device), output_prob], dim=1)
        return output_prob

    def generate(self, z):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        z_mem, z_emb = self.deal_z(z)
        total_sentence_probs = self.generate_one_sentence(input_idx=None, z=z_mem, z_emb=z_emb)
        output_idx = torch.argmax(total_sentence_probs, dim=2)  # [N, max_len-1]
        return output_idx, total_sentence_probs

    def forward(self, inputs, z, pad_mask=None, is_training=True, easy=True):
        '''
        inputs: [N, S]
        z: [N, dist_size + dispersing]
        N: batch_size
        S: sentence length
        E: emb_size
        '''
        if is_training and self.training:
            if pad_mask is None:
                tgt_key_padding_mask = None
            else:
                tgt_key_padding_mask = (pad_mask == 0).to(self.device)  # N x S pad_mask.to(inputs.device)
                tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            # if easy mode
            z_mem, z_emb = self.deal_z(z)
            inputs = inputs[:, :-1]
            output_prob = self.generate_one_sentence(input_idx=inputs, z=z_mem, z_emb=z_emb,
                                                     tgt_key_padding_mask=tgt_key_padding_mask)
            output = output_prob
        else:
            output, _ = self.generate(z)
        return output

