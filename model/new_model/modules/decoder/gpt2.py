#!/usr/bin/python


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers import BartTokenizer, BartModel
from model.new_model.utils.gpt2.mygpt2 import GPT2ForLatentConnector

class NewGPT(nn.Module):
    def __init__(self, config, emb_layer=None, encoder=None, **kwargs): 
        super().__init__()
        gpt_config = config.gpt_config
        self.config = config
        # load models
        self.tokenizer = AutoTokenizer.from_pretrained(gpt_config.gpt2_path)
        self.model = GPT2ForLatentConnector.from_pretrained(gpt_config.gpt2_path, latent_size=config.decoder_input_size)
        trained_list = ['h.' + str(i) for i in range(6, 12)]
        # trained_list = ['h.' + str(i) for i in range(0, 12)]
        
        for name, param in self.named_parameters():    
            for trained_layer in trained_list:
                if trained_layer in name:
                    print(name, 'need train')
                    break
                else:
                    continue
            else:
                if 'past_proj_linear' not in name and 'ln_f' not in name:
                # if 'past_proj_linear' not in name:
                    print(name, 'dont need train')
                    param.requires_grad = False
                else:
                    print(name, 'need train')

        # embedding layer
        self.pad_idx = gpt_config.pad_idx
        self.eos_idx = gpt_config.eos_idx
        self.bos_idx = gpt_config.bos_idx
        self.mask_idx = gpt_config.mask_idx
        # other setting
        self.max_len = config.seq_len

        if self.config.decoder_input_size != self.config.encoder_output_size:
            self.proj_layer = nn.Linear(self.config.encoder_output_size, self.config.decoder_input_size, bias=False)


    def forward(self, sentence=None, memory=None, encoding_indices=None, generate=False, label_text=None, **kwargs):
        if self.config.decoder_input_size != self.config.encoder_output_size:
            memory = self.proj_layer(memory)
        if generate is False:
            assert sentence is not None, 'sentence is None'
            # 训练过程，先tokenize句子，然后加上memory一起重建之
            sentence_idx, attention_mask = self.tokenize(sentence, device=memory.device)
            attention_mask = torch.cat([torch.ones(memory.shape[0], memory.shape[1]).to(memory.device), attention_mask], dim=1)
            outputs = self.model(sentence_idx,
                                 past=memory,
                                 labels=sentence_idx,
                                 attention_mask=attention_mask,
                                 label_ignore=self.pad_idx)
            # print('loss is', outputs[0], outputs[0].shape)
            sentences = self.rebuild_from_logits(outputs[1])
            result_dict = {
                "loss_reconstruct": torch.sum(outputs[0]),
                "pred_idx": torch.argmax(outputs[1], dim=2).detach(),
                'logprobs': outputs[1],
                'sentences': sentences
            }
        else:
            pass
            # if label_text is None:
            #     result_dict = self.generate(memory=memory)
            # else:
            #     result_dict = self.generate_by_label(memory=memory, label_text=label_text)
        return result_dict

    def generate(self, memory, repitition_penalty=1.0, temperature=1.0, top_k=1, top_p=0.0):
        result_dict = {'pred_idx': [], 'sentences': []}
        for i in range(memory.shape[0]):
            context_tokens = self.tokenizer.encode('<BOS>')
            out = self.sample_sequence_conditional(
                context=context_tokens,
                past=memory[i, :, :].unsqueeze(0),
                temperature=temperature,
                device=memory.device,
                top_k=top_k,
                top_p=top_p,
            )
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            #result_dict['pred_idx'].append(out)
            result_dict['sentences'].append(text_x1)
        #result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        return result_dict
    

    def generate_by_text(self, texts, memory, repitition_penalty=1.0, temperature=1.0, top_k=5, top_p=0.0):
        """向外生成句子的接口

        Args:
            texts (_type_): _description_
            memory (_type_): _description_
            repitition_penalty (float, optional): _description_. Defaults to 1.0.
            temperature (float, optional): _description_. Defaults to 1.0.
            top_k (int, optional): _description_. Defaults to 5.
            top_p (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        result_dict = {'pred_idx': [], 'sentences': []}
        for i in range(len(texts)):
            context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(texts[i]))
            context = torch.tensor(context, dtype=torch.long, device=memory.device)
            out = self.sample_sequence_conditional(
                context=context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                label_text=None,
                device=memory.device
            )
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            #result_dict['pred_idx'].append(out)
            result_dict['sentences'].append(text_x1)
        #result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        return result_dict

    def sample_sequence_conditional(self, context, past=None, device=None, num_samples=1, temperature=1.0, top_k=2, top_p=0.0, label_text=None):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        if label_text is None:
            with torch.no_grad():
                #while True:
                i = 0
                while i < self.max_len:
                    inputs = {'input_ids': generated}
                    outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                    next_token_logits = outputs[0][0, -1, :] / temperature
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                    # pdb.set_trace()
                    if next_token.unsqueeze(0)[0,0].item() == self.tokenizer.encode('<EOS>')[0]:
                        break
                    i+=1
        else:
            print(label_text)
            label_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + label_text))
            label_idx = torch.tensor(label_idx, dtype=torch.long, device=device)
            print(label_idx)
            label_first_word_idx = label_idx[0]
            with torch.no_grad():
                #while True:
                i = 0
                label_flag = 0
                while i < self.max_len:
                    inputs = {'input_ids': generated}
                    outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                    next_token_logits = outputs[0][0, -1, :] / temperature
                    prob_value = next_token_logits[label_first_word_idx].item()
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    filtered_logits[label_first_word_idx] = prob_value
  
                    tokens_prob = F.softmax(filtered_logits, dim=-1)
                    label_prob = 1/(1+np.exp(-tokens_prob[label_first_word_idx]))  / 5
                    # print(label_prob)
                    if label_flag == 0 and np.random.rand(1)[0] < label_prob:
                        next_token = label_idx
                        i+= len(label_idx)
                        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                        label_flag = 1
                    else:
                        next_token = torch.multinomial(tokens_prob, num_samples=1)
                        #print('next token is', self.tokenizer.decode(next_token))
                        i+=1
                        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                    # pdb.set_trace()
                    if next_token.unsqueeze(0)[0,0].item() == self.tokenizer.encode('<EOS>')[0]:
                        break
                    # print(generated)
                    

        return generated

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def rebuild_sentence(self, input_idx):
        sentences = []
        if len(input_idx.shape) > 1:
            for i in range(input_idx.shape[0]):
                sentence = self.tokenizer.decode(input_idx[i, :].tolist(), clean_up_tokenization_spaces=True)
                sentence = sentence.split()
                sentence = ' '.join(sentence)
                # sentence = self.tokenizer.convert_ids_to_tokens()
                sentences.append(sentence)
        else:
            # sentence = self.tokenizer.convert_ids_to_tokens(input_idx)
            sentence = self.tokenizer.decode(input_idx.tolist(), clean_up_tokenization_spaces=True)
            sentences.append(sentence)
        return sentences

    def rebuild_from_logits(self, logits):
        input_idx = logits.argmax(dim=2)
        sentence = self.rebuild_sentence(input_idx)
        return sentence

    def tokenize(self, sentence, device):
        output = self.tokenizer(sentence,
                                add_special_tokens=True,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_len,
                                return_tensors="pt")
        idx = output['input_ids']
        idx = idx.to(device)
        attn_mask = output['attention_mask'].to(device)
        return idx, attn_mask


class NewGPT_Old(nn.Module):
    def __init__(self, config, emb_layer=None, encoder=None, **kwargs): 
        super().__init__()
        gpt_config = config.gpt_config
        self.config = config
        # load models
        self.tokenizer = AutoTokenizer.from_pretrained(gpt_config.gpt2_path)
        self.model = GPT2ForLatentConnector.from_pretrained(gpt_config.gpt2_path)
        # trained_list = ['h.' + str(i) for i in range(6, 12)]
        
        # for name, param in self.named_parameters():    
        #     for trained_layer in trained_list:
        #         if trained_layer in name:
        #             print(name, 'need train')
        #             break
        #         else:
        #             continue
        #     else:
        #         if 'past_proj_linear' not in name and 'ln_f' not in name:
        #         # if 'past_proj_linear' not in name:
        #             print(name, 'dont need train')
        #             param.requires_grad = False
        #         else:
        #             print(name, 'need train')

        # embedding layer
        self.pad_idx = gpt_config.pad_idx
        self.eos_idx = gpt_config.eos_idx
        self.bos_idx = gpt_config.bos_idx
        self.mask_idx = gpt_config.mask_idx
        # other setting
        self.max_len = config.seq_len

        if self.config.decoder_input_size != self.config.encoder_output_size:
            self.proj_layer = nn.Linear(self.config.encoder_output_size, self.config.decoder_input_size, bias=False)


    def forward(self, sentence=None, memory=None, encoding_indices=None, generate=False, label_text=None, **kwargs):
        if self.config.decoder_input_size != self.config.encoder_output_size:
            memory = self.proj_layer(memory)
        if generate is False:
            assert sentence is not None, 'sentence is None'
            # 训练过程，先tokenize句子，然后加上memory一起重建之
            sentence_idx, attention_mask = self.tokenize(sentence, device=memory.device)
            attention_mask = torch.cat([torch.ones(memory.shape[0], memory.shape[1]).to(memory.device), attention_mask], dim=1)
            outputs = self.model(sentence_idx,
                                 past=memory,
                                 labels=sentence_idx,
                                 attention_mask=attention_mask,
                                 label_ignore=self.pad_idx)
            # print('loss is', outputs[0], outputs[0].shape)
            sentences = self.rebuild_from_logits(outputs[1])
            result_dict = {
                "loss_reconstruct": torch.sum(outputs[0]),
                "pred_idx": torch.argmax(outputs[1], dim=2).detach(),
                'logprobs': outputs[1],
                'sentences': sentences
            }
        else:
            if label_text is None:
                result_dict = self.generate(memory=memory)
            else:
                result_dict = self.generate_by_label(memory=memory, label_text=label_text)
        return result_dict

    def generate(self, memory, repitition_penalty=1.0, temperature=1.0, top_k=1, top_p=0.0):
        result_dict = {'pred_idx': [], 'sentences': []}
        for i in range(memory.shape[0]):
            context_tokens = self.tokenizer.encode('<BOS>')
            out = self.sample_sequence_conditional(
                context=context_tokens,
                past=memory[i, :, :].unsqueeze(0),
                temperature=temperature,
                device=memory.device,
                top_k=top_k,
                top_p=top_p,
            )
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            #result_dict['pred_idx'].append(out)
            result_dict['sentences'].append(text_x1)
        #result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        return result_dict
    

    def generate_by_text(self, texts, memory, repitition_penalty=1.0, temperature=1.0, top_k=5, top_p=0.0):
        """向外生成句子的接口

        Args:
            texts (_type_): _description_
            memory (_type_): _description_
            repitition_penalty (float, optional): _description_. Defaults to 1.0.
            temperature (float, optional): _description_. Defaults to 1.0.
            top_k (int, optional): _description_. Defaults to 5.
            top_p (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        
        if self.config.decoder_input_size != self.config.encoder_output_size:
            memory = self.proj_layer(memory)
        result_dict = {'pred_idx': [], 'sentences': []}
        assert len(texts) == memory.shape[0], 'texts shape %d is not equal to memory shape %d' % (len(texts), memory.shape[0])

        for i in range(len(texts)):
            context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(texts[i]))
            print('generate by text', texts[i], context)
            context = torch.tensor(context, dtype=torch.long, device=memory.device)
            out = self.sample_sequence_conditional(
                context=context,
                temperature=temperature,
                past=memory[i, :, :].unsqueeze(0),
                top_k=top_k,
                top_p=top_p,
                label_text=None
            )
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            #result_dict['pred_idx'].append(out)
            result_dict['sentences'].append(text_x1)
        #result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        return result_dict


    def generate_by_label(self, memory, label_text: list, repitition_penalty=1.0, temperature=1.0, top_k=1, top_p=0.0):
        result_dict = {'pred_idx': [], 'sentences': []}
        for i in range(memory.shape[0]):
            context_tokens = self.tokenizer.encode('<BOS>')
            out = self.sample_sequence_conditional(
                context=context_tokens,
                past=memory[i, :, :].unsqueeze(0),
                temperature=temperature,
                device=memory.device,
                top_k=top_k,
                top_p=top_p,
                label_text=label_text[i],
            )
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            #result_dict['pred_idx'].append(out)
            result_dict['sentences'].append(text_x1)
        #result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        return result_dict

    def sample_sequence_conditional(self, context, past=None, device=None, num_samples=1, temperature=1.0, top_k=5, top_p=0.0, label_text=None):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        if label_text is None:
            with torch.no_grad():
                #while True:
                i = 0
                while i < self.max_len:
                    inputs = {'input_ids': generated, 'past': past}
                    outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                    next_token_logits = outputs[0][0, -1, :] / temperature
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                    # pdb.set_trace()
                    if next_token.unsqueeze(0)[0,0].item() == self.tokenizer.encode('<EOS>')[0]:
                        break
                    i+=1
        else:
            label_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + label_text))
            label_idx = torch.tensor(label_idx, dtype=torch.long, device=past.device)
            label_first_word_idx = label_idx[0]
            with torch.no_grad():
                #while True:
                i = 0
                label_flag = 0
                while i < self.max_len:
                    inputs = {'input_ids': generated}
                    outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                    next_token_logits = outputs[0][0, -1, :] / temperature
                    prob_value = next_token_logits[label_first_word_idx].item()
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    filtered_logits[label_first_word_idx] = prob_value
  
                    tokens_prob = F.softmax(filtered_logits, dim=-1)
                    label_prob = 1/(1+np.exp(-tokens_prob[label_first_word_idx].cpu().item())) / 3
                    # print(label_prob)
                    if label_flag == 0 and np.random.rand(1)[0] < label_prob:
                        next_token = label_idx
                        i+= len(label_idx)
                        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                        label_flag = 1
                    else:
                        next_token = torch.multinomial(tokens_prob, num_samples=1)
                        #print('next token is', self.tokenizer.decode(next_token))
                        i+=1
                        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                    # pdb.set_trace()
                    if next_token.unsqueeze(0)[0,0].item() == self.tokenizer.encode('<EOS>')[0]:
                        break
                    # print(generated)
                    
        return generated

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def rebuild_sentence(self, input_idx):
        sentences = []
        if len(input_idx.shape) > 1:
            for i in range(input_idx.shape[0]):
                sentence = self.tokenizer.decode(input_idx[i, :].tolist(), clean_up_tokenization_spaces=True)
                sentence = sentence.split()
                sentence = ' '.join(sentence)
                # sentence = self.tokenizer.convert_ids_to_tokens()
                sentences.append(sentence)
        else:
            # sentence = self.tokenizer.convert_ids_to_tokens(input_idx)
            sentence = self.tokenizer.decode(input_idx.tolist(), clean_up_tokenization_spaces=True)
            sentences.append(sentence)
        return sentences

    def rebuild_from_logits(self, logits):
        input_idx = logits.argmax(dim=2)
        sentence = self.rebuild_sentence(input_idx)
        return sentence

    def tokenize(self, sentence, device):
        output = self.tokenizer(sentence,
                                add_special_tokens=True,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_len,
                                return_tensors="pt")
        idx = output['input_ids']
        idx = idx.to(device)
        attn_mask = output['attention_mask'].to(device)
        return idx, attn_mask