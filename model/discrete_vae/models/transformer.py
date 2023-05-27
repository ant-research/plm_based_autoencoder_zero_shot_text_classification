
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import logging as log
from model.discrete_vae.models.simple_module import (
    PositionalEncoding,
    Pooler,
)
from model.discrete_vae.models.vq_quantizer import DVQ
from model.discrete_vae.models.gs_quantizer import ConcreteQuantizer


class TransformerQuantizerEncoder(nn.Module):
    """transformer encoder + vq

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config, emb_layer, **kwargs):
        super().__init__()
        self.use_memory = config.use_memory
        self.hid_emb = config.emb_size
        self.dropout = config.dropout_rate
        self.num_heads = config.num_heads
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        
        self.pad_idx = emb_layer.emb_model.pad_idx
        self.num_embeddings = emb_layer.emb_model.get_vocab_len()

        self.embedding = nn.Embedding(
            self.num_embeddings, self.hid_emb, padding_idx=self.pad_idx
        )
        self.pos_encoder = PositionalEncoding(
            d_model=self.hid_emb, dropout=self.dropout
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

        # decompose
        split = config.decompose_number
        D = self.hid_emb

        # specific to quantizer
        if config.quantizer_type == "vq":
            self.quantizer = DVQ(
                config,
                num_embeddings=config.disper_num,
                embedding_dim=D,
                split=split,
                decompose_option="slice",
            )
        elif config.quantizer_type == "concrete":
            self.quantizer = ConcreteQuantizer(
                config, num_embeddings=config.disper_num, embedding_dim=D, split=split
            )


        # sentence
        if config.quantizer_type == "vq":
            d_proj = config.decompose_number * self.hid_emb
        elif config.quantizer_type == "concrete":
            d_proj = config.decompose_number * config.disper_num
        else:
            raise KeyError
        self.pooler = Pooler(
            project=True,
            pool_type="mean",
            d_inp=self.hid_emb,
            d_proj=d_proj,
        )

        self.output_dim = self.hid_emb

    def forward(self, src):
        src_pad_mask = src == self.pad_idx
        src_nopad_mask = src != self.pad_idx
        nopad_lengths = src_nopad_mask.sum(dim=-1).long()

        src_emb = self.embedding(src).transpose(0, 1)
        src_emb = self.pos_encoder(src_emb)
        src_mask = None

        memory = self.encoder(
            src_emb, src_key_padding_mask=src_pad_mask, mask=src_mask
        ).transpose(0, 1)


        # seq2vec: mean pooling to get sentence representation
        # bsz × (M * D) or bsz × (M * K)
        pooled_memory = self.pooler(memory, src_nopad_mask)
        print('pooled memory is', pooled_memory)
        quantizer_out = self.quantizer(pooled_memory)
        # mask is filled with 1, bsz × M
        # src_nopad_mask = torch.ones_like(quantizer_out['encoding_indices'])
        if self.use_memory == True:
            quantizer_out['quantized_stack'] = memory
        else:
            src_nopad_mask = quantizer_out["encoding_indices"] != -1
        # bsz × M × D
        # quantizer_out['quantized_stack'] = memory
        enc_out = quantizer_out["quantized_stack"]
        

        return {
            "quantizer_out": quantizer_out,
            "nopad_mask": src_nopad_mask,
            "sequence": enc_out,
        }

    def get_output_dim(self):
        return self.output_dim


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


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config, encoder, emb_layer):
        super().__init__()

        self.encoder = encoder
        self.easy_decoder = config.easy_decoder
        
        self.hid_emb = config.emb_size
        self.dropout = config.dropout_rate
        self.num_heads = config.num_heads
        self.forward_expansion = config.forward_expansion
        self.num_layers = config.num_layers
        self.pad_idx = emb_layer.emb_model.pad_idx
        self.eos_idx = emb_layer.emb_model.eos_idx
        self.bos_idx = emb_layer.emb_model.bos_idx
        self.max_len = config.seq_len
        self.num_embeddings = emb_layer.emb_model.get_vocab_len()
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hid_emb,
            nhead=self.num_heads,
            dim_feedforward=self.forward_expansion,
            dropout=self.dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=config.num_layers
        )

        self.num_embeddings = self.num_embeddings
        # self.first_word_proj_layer = nn.Linear(self.hid_emb * config.decompose_number, self.hid_emb)
        self.classifier = nn.Sequential(
            nn.Linear(self.hid_emb, self.num_embeddings),
            nn.LogSoftmax(dim=-1),
        )

        self.decoding_util = DecodingUtil(self.num_embeddings, self.pad_idx)
        self.init_weights()
        
        self.kl_fbp = config.concrete_kl_fbp_threshold
        self.kl_beta = config.concrete_kl_beta

    def forward(self, inputs_batch, pad_mask=None):
        input = input_from_batch(inputs_batch, self.eos_idx, self.pad_idx)
        src = input['enc_in']

        # encoder
        enc_outdict = self.encoder(src)
        memory = enc_outdict["sequence"].transpose(0, 1)
        return self.decode(input, memory, enc_outdict)

    def generate_square_subsequent_mask(self, size):
        # Generate mask covering the top right triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_one_word(self, input, enc_outdict, memory):
        tgt = input
        tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
        tgt_emb = self.encoder.pos_encoder(tgt_emb)

        tgt_pad_mask = tgt == self.pad_idx

        # causal masking
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.shape[0]).to(tgt_emb.device)
        src_nopad_mask = enc_outdict["nopad_mask"]
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
        logprobs = self.classifier(output)

        return logprobs

    def get_last_word(self, output_prob):
        '''
        output_prob: [Num, seq_len, num_tokens]
        '''
        next_prob = output_prob[:, -1].unsqueeze(1)  # [N, 1, num_tokens]
        output_idx = torch.argmax(output_prob, dim=2)[:, -1]  # [N, max_len-1]
        return next_prob, output_idx
    
    def generate(self, enc_outdict, memory):
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
            output_prob = self.generate_one_word(next_input, enc_outdict, memory)
            # 获取最后一个词的概率以及max idx
            next_prob, output_idx = self.get_last_word(output_prob)
            # 最大的那个概率的词放入
            next_input = torch.cat([next_input, output_idx.unsqueeze(1)], dim=1)  # [N, now_len+1]
            # 储存概率
            total_sentence_probs.append(next_prob)
        total_sentence_probs = torch.cat(total_sentence_probs, dim=1)
        return total_sentence_probs

    def decode(self, input, memory, enc_outdict):

        # teacher forcing

        dec_out_gold = input["dec_out_gold"]
        if self.easy_decoder is True:
            tgt = input["dec_in"]
            tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
            tgt_emb = self.encoder.pos_encoder(tgt_emb)

            tgt_pad_mask = tgt == self.pad_idx

            # causal masking
            print('memory', memory.shape)
            tgt_mask = self.generate_square_subsequent_mask(len(tgt_emb)).to(tgt_emb.device)

            src_nopad_mask = enc_outdict["nopad_mask"]
            src_pad_mask = src_nopad_mask == 0
            output = self.decoder_layer(
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
            logprobs = self.generate(enc_outdict=enc_outdict, memory=memory)

        dec_outdict = self.decoding_util.forward(logprobs, dec_out_gold)
        loss_reconstruct = dec_outdict["loss"]
        pred_idx = dec_outdict["pred_idx"]
        ntokens = dec_outdict["ntokens"]

        # total loss
        quantizer_out = enc_outdict["quantizer_out"]

        if type(self.encoder.quantizer) in [DVQ]:
            loss = loss_reconstruct + quantizer_out["loss"]
            result = {
                "loss_commit": quantizer_out["loss_commit"],
            }
        elif type(self.encoder.quantizer) == ConcreteQuantizer:
            actual_kl = quantizer_out["kl"]
            if self.training:
                # apply thershold to kl (batch mean), actual_kl is sum
                # fbp_kl = torch.clamp(actual_kl, min=self.kl_fbp * bsz)
                # loss = loss_reconstruct + fbp_kl
                if actual_kl < (self.kl_fbp * logprobs.shape[0]):
                    loss = loss_reconstruct
                else:
                    loss = loss_reconstruct + self.kl_beta * actual_kl
            else:
                loss = loss_reconstruct

            result = {
                "kl": actual_kl,
            }
        else:
            raise KeyError

        result.update(
            {
                "indices": quantizer_out["encoding_indices"].detach(),
                "loss_reconstruct": loss_reconstruct.detach(),
                "loss": loss,
                "pred_idx": pred_idx.detach(),
                # "ntokens": ntokens,
            }
        )
        return result

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    
def input_from_batch(inputs, eos_idx, pad_idx):
    sent = inputs  # shape (batch_size, seq_len)
    batch_size, seq_len = sent.size()

    # no <SOS> and <EOS>
    enc_in = sent[:, 1:-1].clone()
    enc_in[enc_in == eos_idx] = pad_idx

    # no <SOS>
    dec_out_gold = sent[:, 1:].contiguous()

    # no <EOS>
    dec_in = sent[:, :-1].clone()
    dec_in[dec_in == eos_idx] = pad_idx

    out = {
        "batch_size": batch_size,
        "dec_in": dec_in,
        "dec_out_gold": dec_out_gold,
        "enc_in": enc_in,
        "sent": sent,
    }
    return out