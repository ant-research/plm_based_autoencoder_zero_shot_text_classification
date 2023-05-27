import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel, GPT2LMHeadModel


class NewGPT(nn.Module):
    def __init__(self, config, device=None, **kwargs): 
        super().__init__()
        gpt_config = config.gpt_config
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(gpt_config.gpt2_path)
        self.model = GPT2LMHeadModel.from_pretrained(gpt_config.gpt2_path)
        self.device = device
        # embedding layer
        self.pad_idx = gpt_config.pad_idx
        self.eos_idx = gpt_config.eos_idx
        self.bos_idx = gpt_config.bos_idx
        self.mask_idx = gpt_config.mask_idx
        # other setting
        self.max_len = config.seq_len

    def generate(self, memory, repitition_penalty=1.0, temperature=1.0, top_k=30, top_p=0.0):
        result_dict = {'pred_idx': []}
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
            print(out.shape)
            print(out)
            result_dict['pred_idx'].append(out)
            
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()[1:-1]
            text_x1 = ' '.join(text_x1)
            
        result_dict['pred_idx'] = torch.cat(result_dict['pred_idx'], dim=0)
        print('final pred idx result', result_dict['pred_idx'])
        return text_x1
    
    def generate_from_token(self, label, label_text, repitition_penalty=1.0, temperature=1.0, top_k=1, top_p=0.0):
        context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label))
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        generated = context
        # outputs = self.model(label_idx, attention_mask=label_attn_mask)
        for _ in range(self.max_len):
            inputs = {'input_ids': generated[0].unsqueeze(0)}
            outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet 
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        text = self.tokenizer.decode(generated[0,:].tolist(), clean_up_tokenization_spaces=True)
        return text

    def generate_by_label(self, label, label_text, repitition_penalty=1.0, temperature=1.0, top_k=5, top_p=0.0):
        result_dict = {'pred_idx': [], 'sentences': []}
        for i in range(len(label)):
            context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label[i]))
            print('input text is:', label[i])
            context = torch.tensor(context, dtype=torch.long, device=self.device)
            out = self.sample_sequence_conditional(
                context=context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                label_text=label_text[i],
            )
            print('out is', out, out.shape)
            text_x1 = self.tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            print('output text is', text_x1)
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
            label_idx = torch.tensor(label_idx, dtype=torch.long, device=self.device)
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
        sentence = self.tokenizer.convert_ids_to_tokens(input_idx)
        return sentence

    def rebuild_from_logits(self, logits):
        input_idx = logits.argmax(dim=1)
        sentence = self.rebuild_sentence(input_idx)
        return sentence

    def tokenize(self, sentence):
        output = self.tokenizer(sentence,
                                add_special_tokens=True,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_len, return_tensors="pt")
        idx = output['input_ids']
        idx = torch.tensor(idx).to(self.model.device)
        # print('生成的idx出来的句子是：', self.rebuild_sentence(idx))
        attn_mask = torch.tensor(output['attention_mask']).to(self.model.device)
        return idx, attn_mask, output