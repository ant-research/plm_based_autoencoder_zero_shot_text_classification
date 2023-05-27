
from numpy import result_type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from model.new_model.modules.simple_modules import PositionalEncoding
import torch.utils.checkpoint as cp


class DecodingUtil(object):
    def __init__(self, vsize, pad_idx):
        self.criterion = nn.NLLLoss(reduction="none")
        self.vsize = vsize
        self.pad_idx = pad_idx

    def forward(self, logprobs, dec_out_gold):
        # print(dec_out_gold.shape)
        # reconstruction loss
        loss_reconstruct = self.criterion(
            logprobs.contiguous().view(-1, self.vsize), dec_out_gold.view(-1)
        )
        # mask out padding
        nopad_mask = (dec_out_gold != self.pad_idx).view(-1).float()
        nll = (loss_reconstruct * nopad_mask).view(logprobs.shape[:-1]).detach()
        loss_reconstruct = (loss_reconstruct * nopad_mask).sum()

        # post-processing
        nopad_mask2 = dec_out_gold != self.pad_idx
        pred_idx = torch.argmax(logprobs, dim=2)
        pred_idx = pred_idx * nopad_mask2.long()

        ntokens = nopad_mask.sum().item()

        return {
            "loss": loss_reconstruct,
            "pred_idx": pred_idx,
            "nopad_mask": nopad_mask2,
            "ntokens": ntokens,
            "nll": nll,
        }
        
class DecodingUtilDistance(object):
    def __init__(self, emb_size, pad_idx, emb_layer):
        self.criterion = nn.MSELoss(reduction="none")
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        self.emb_layer = emb_layer

    def forward(self, logprobs, dec_out_gold):
        # print(dec_out_gold.shape)
        # reconstruction loss
        dec_out_emb = self.emb_layer(dec_out_gold)
        #loss_reconstruct = self.criterion(
        #    logprobs.contiguous(), dec_out_emb
        #)
        
        # mask out padding
        nopad_mask = (dec_out_gold != self.pad_idx).to(logprobs.device)
        nopad_mask = nopad_mask.unsqueeze(2)
        x = torch.masked_select(dec_out_emb, nopad_mask)  # [batch_size, seq_len-1, emb_size]
        recon_x = torch.masked_select(logprobs.contiguous(), nopad_mask)
        loss_reconstruct = F.mse_loss(recon_x, x)

        # post-processing
        nopad_mask2 = dec_out_gold != self.pad_idx
        pred_idx = torch.argmax(logprobs, dim=2)
        pred_idx = pred_idx * nopad_mask2.long()

        ntokens = nopad_mask.sum().item()

        return {
            "loss": loss_reconstruct,
            "pred_idx": pred_idx,
            "nopad_mask": nopad_mask2,
            "ntokens": ntokens,
        }


class TransformerDecoderBase(nn.Module):
    def __init__(self, config, emb_layer=None, encoder=None, **kwargs) -> None:
        super().__init__()
        
        # embedding layer
        self.pad_idx = emb_layer.emb_model.pad_idx
        self.eos_idx = emb_layer.emb_model.eos_idx
        self.bos_idx = emb_layer.emb_model.bos_idx

        # training setting
        self.easy_decoder = config.easy_decoder
        self.latent_size = config.decoder_input_size
        self.hid_emb = config.emb_size
        self.dropout = config.dropout_rate
        self.num_heads = config.num_heads
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        self.max_len = config.seq_len
        self.num_embeddings = emb_layer.emb_model.get_vocab_len()
        
        self.encoder = encoder
        
        # latent variable over layer
        self.latent_layer = nn.Linear(self.latent_size, self.hid_emb)
        
    def generate_square_subsequent_mask(self, size):
        # Generate mask covering the top right  triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def get_last_word(self, output_prob):
        '''
        output_prob: [Num, seq_len, num_tokens]
        '''
        next_prob = output_prob[:, -1].unsqueeze(1)  # [N, 1, num_tokens]
        output_idx = torch.argmax(output_prob, dim=2)[:, -1]  # [N, max_len-1]
        return next_prob, output_idx


class TransformerDecoder(TransformerDecoderBase):
    def __init__(self, config, decoder_name='transformer decoder', emb_layer=None, encoder=None, **kwargs):
        super().__init__(config, emb_layer=emb_layer, encoder=encoder)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hid_emb,
            nhead=self.num_heads,
            dim_feedforward=self.forward_expansion,
            dropout=self.dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=config.num_layers
        )

        self.num_embeddings = self.num_embeddings
        # self.first_word_proj_layer = nn.Linear(self.hid_emb * config.decompose_number, self.hid_emb)
        self.classifier = nn.Sequential(
            nn.Linear(self.hid_emb, self.num_embeddings),
            nn.LogSoftmax(dim=-1),
        )

        self.decoding_util = DecodingUtil(self.num_embeddings, self.pad_idx)
        self.init_weights()


    def forward(self, inputs, memory, encoding_indices=None, generate=False, **kwargs):
        result_dict = self.decode(input=inputs,
                                  memory=memory,
                                  encoding_indices=encoding_indices,
                                  generate=generate)
    
        return result_dict
    
    def generate_one_word(self, input, src_nopad_mask, memory):
        print('hard memory, shape is ', input.shape, torch.cuda.mem_get_info())
        tgt = input
        tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
        tgt_emb = self.encoder.pos_encoder(tgt_emb)

        tgt_pad_mask = tgt == self.pad_idx

        # causal masking
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.shape[0]).to(tgt_emb.device)
        src_nopad_mask = src_nopad_mask
        src_pad_mask = src_nopad_mask == 0
        # print(src_nopad_mask, src_nopad_mask.shape)
        # print(src_pad_mask, src_pad_mask.shape)
        output = self.decoder(
            tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )
        # output = cp.checkpoint(self.decoder, tgt_emb,
        #     memory, tgt_mask, None,
        #     tgt_pad_mask,
        #     src_pad_mask,
        # )
        output = output.transpose(0, 1)

        # classifier
        logprobs = self.classifier(output)
        # logprobs = cp.checkpoint(self.classifier, output)

        return logprobs
    
    def generate(self, src_nopad_mask, memory):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        # first word bos
        next_input = torch.ones(memory.shape[1], 1).fill_(self.bos_idx).type(torch.long).to(memory.device)  # [N, 1]
        total_sentence_probs = []
        
        # start bos prob
        for _ in range(self.max_len-1):
            # 从input生成
            output_prob = self.generate_one_word(next_input, src_nopad_mask, memory)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
            torch.cuda.empty_cache()
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return total_sentence_probs

    def decode(self, input, memory, encoding_indices, generate=False):
        # teacher forcing
        memory = self.latent_layer(memory)
        memory = memory.transpose(0, 1)
        # add t encoding indices
        if encoding_indices is None:
            encoding_indices = torch.zeros(memory.shape[1], memory.shape[0]).long().to(memory.device)
        else:
            distil_indices = torch.zeros(encoding_indices.shape[0],
                                        memory.shape[0]-encoding_indices.shape[1]).long().to(encoding_indices.device)
            encoding_indices = torch.cat([distil_indices, encoding_indices], dim=1)
        src_nopad_mask = encoding_indices != -1
        if self.easy_decoder is True and generate is False:
            # fast autoregressive
            tgt = input["dec_in"]
            tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
            tgt_emb = self.encoder.pos_encoder(tgt_emb)

            tgt_pad_mask = tgt == self.pad_idx

            # causal masking
            print('memory', memory.shape)
            tgt_mask = self.generate_square_subsequent_mask(len(tgt_emb)).to(tgt_emb.device)

            src_pad_mask = src_nopad_mask == 0
            output = self.decoder(
                tgt_emb,
                memory=memory,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
                tgt_mask=tgt_mask,
            )

            output = output.transpose(0, 1)

            # classifier
            logprobs = self.classifier(output)
        else:
            # auto regressive without true label
            logprobs = self.generate(src_nopad_mask=src_nopad_mask, memory=memory)
            # logprobs = cp.checkpoint(self.generate, src_nopad_mask, memory)

        # 判断是否是生成模型，否的话算loss，是的话只算pred idx
        if generate is False:
            assert input is not None
            dec_out_gold = input["dec_out_gold"]
            dec_outdict = self.decoding_util.forward(logprobs, dec_out_gold)
            loss_reconstruct = dec_outdict["loss"]
            pred_idx = dec_outdict["pred_idx"]
            # ntokens = dec_outdict["ntokens"]
        else:
            pred_idx = torch.argmax(logprobs, dim=2)
            loss_reconstruct = None

        result = {
            "loss_reconstruct": loss_reconstruct,
            "pred_idx": pred_idx.detach(),
            'logprobs': logprobs
            # "ntokens": ntokens,
        }
        return result


class TransformerDecoderDistance(TransformerDecoderBase):
    """
    Transformer but use cos similarity as output probability
    两种算法：
    1 计算cross entropy loss， 使用autoregressive cos similarity算出来的probability，
    2 计算mse loss，使用autoregssive transformer出来的结果算loss，使用cos similarity算idx

    Args:
        TransformerDecoderBase (_type_): _description_
    """    
    def __init__(self, config, decoder_name='transformer decoder', emb_layer=None, encoder=None, **kwargs):
        super().__init__(config, emb_layer=emb_layer, encoder=encoder)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hid_emb,
            nhead=self.num_heads,
            dim_feedforward=self.forward_expansion,
            dropout=self.dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=config.num_layers
        )

        self.num_embeddings = self.num_embeddings
        # self.first_word_proj_layer = nn.Linear(self.hid_emb * config.decompose_number, self.hid_emb)
        self.emb_weight = emb_layer.emb_model.embedding_layer.weight.detach()
        self.softmax_layer = nn.LogSoftmax(dim=-1)
        self.decoding_util = DecodingUtil(self.num_embeddings, self.pad_idx)
        # self.decoding_util = DecodingUtilDistance(self.hid_emb, self.pad_idx, self.encoder.embedding)
        self.init_weights()


    def forward(self, inputs, memory, encoding_indices=None, generate=False, **kwargs):
        result_dict = self.decode(input=inputs,
                                  memory=memory,
                                  encoding_indices=encoding_indices,
                                  generate=generate)
    
        return result_dict
    
    def match_network_ed(self, output):
        """
        利用Euclidean distance来计算probability

        Args:
            output (_type_): _description_

        Returns:
            _type_: _description_
        """        
        sim_output_list = []
        emb_weight = self.emb_weight.detach().to(output.device)
        for i in range(output.shape[0]):
            sim_vec = output[i, -1, :].reshape(1, 1, -1) #只计算最后一个词的概率
            sim = torch.cdist(sim_vec, emb_weight)
            sim = sim.reshape(1, 1, -1)  # [1, S, NTokens]
            sim_output_list.append(sim)
        sim_output = torch.cat(sim_output_list, dim=0)
        sim_output = self.softmax_layer(-sim_output)
        # sim_output = torch.cdist(output, self.emb_weight)
        return sim_output

    def generate_one_word(self, input, src_nopad_mask, memory):
        tgt = input
        tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
        tgt_emb = self.encoder.pos_encoder(tgt_emb)

        tgt_pad_mask = tgt == self.pad_idx

        # causal masking
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.shape[0]).to(tgt_emb.device)
        src_nopad_mask = src_nopad_mask
        src_pad_mask = src_nopad_mask == 0
        # print(src_nopad_mask, src_nopad_mask.shape)
        # print(src_pad_mask, src_pad_mask.shape)
        output = self.decoder(
            tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )
        output = output.transpose(0, 1)

        # classifier
        logprobs = self.match_network_ed(output)

        return logprobs
    
    def generate(self, src_nopad_mask, memory):
        '''
        inputs: z latent varibale
        outputs:
            next_input: [N, seq_len] reconstruction sentence idx
            total_sentence_probs [N, seq_len, ntokens] sentence probability
        '''
        
        # first word bos
        next_input = torch.ones(memory.shape[1], 1).fill_(self.bos_idx).type(torch.long).to(memory.device)  # [N, 1]
        total_sentence_probs = []
        # start bos prob
        for _ in range(self.max_len-1):
            # 从input生成
            output_prob = self.generate_one_word(next_input, src_nopad_mask, memory)
            # 获取最后一个词的概率以及max idx
            # print('output prob', output_prob.shape, output_prob)
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return total_sentence_probs

    def decode(self, input, memory, encoding_indices, generate=False):
        
        # teacher forcing
        memory = self.latent_layer(memory)
        memory = memory.transpose(0, 1)
        # add t encoding indices
        if encoding_indices is None:
            encoding_indices = torch.zeros(memory.shape[1], memory.shape[0]).long().to(memory.device)
        else:
            distil_indices = torch.zeros(encoding_indices.shape[0],
                                        memory.shape[0]-encoding_indices.shape[1]).long().to(encoding_indices.device)
            encoding_indices = torch.cat([distil_indices, encoding_indices], dim=1)
        src_nopad_mask = encoding_indices != -1
        if self.easy_decoder is True:
            # fast autoregressive
            tgt = input["dec_in"]
            tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
            tgt_emb = self.encoder.pos_encoder(tgt_emb)

            tgt_pad_mask = tgt == self.pad_idx

            # causal masking
            # print('memory', memory.shape)
            tgt_mask = self.generate_square_subsequent_mask(len(tgt_emb)).to(tgt_emb.device)

            src_pad_mask = src_nopad_mask == 0
            output = self.decoder(
                tgt_emb,
                memory=memory,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
                tgt_mask=tgt_mask,
            )

            output = output.transpose(0, 1)

            # classifier
            logprobs = self.match_network_ed(output)
        else:
            # auto regressive without true label
            logprobs = self.generate(src_nopad_mask=src_nopad_mask, memory=memory)

        # 判断是否是生成模型，否的话算loss，是的话只算pred idx
        if generate is False:
            assert input is not None
            dec_out_gold = input["dec_out_gold"]
            dec_outdict = self.decoding_util.forward(logprobs, dec_out_gold)
            loss_reconstruct = dec_outdict["loss"]
            pred_idx = dec_outdict["pred_idx"]
            # ntokens = dec_outdict["ntokens"]
        else:
            pred_idx = torch.argmax(logprobs, dim=2)
            loss_reconstruct = None

        result = {
            "loss_reconstruct": loss_reconstruct,
            "pred_idx": pred_idx.detach(),
            'logprobs': logprobs
            # "ntokens": ntokens,
        }
        return result
    
