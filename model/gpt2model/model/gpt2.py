import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model


class MyGPTGenerate(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.max_len = config.seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained('./pretrained_model/gpt2')
        self.model = GPT2Model.from_pretrained('./pretrained_model/gpt2')
        self.device = device

    def forward(self, label):
        label_idx, label_attn_mask, output = self.tokenize(label)

        outputs = self.model(label_idx, attention_mask=label_attn_mask)
        logits = outputs.last_hidden_state
        result_dict = {
            'logits' : logits,
            'label_attn_mask': label_attn_mask
        }
        return result_dict
    
    def generate_from_token(self, label, repitition_penalty=1.0, temperature=1.0, top_k=1, top_p=0.0):
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

    def sample_sequence_conditional(self, context, past=None, device=None, num_samples=1, temperature=1.0, top_k=0, top_p=0.0):
        
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            #while True:
            for _ in range(self.max_len):
                inputs = {'input_ids': generated, 'past': past}
                outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

                # pdb.set_trace()
                # if next_token.unsqueeze(0)[0,0].item() == self.tokenizer.encode('<EOS>')[0]:
                #     break

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
