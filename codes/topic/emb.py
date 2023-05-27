import os
import torch
import torch.nn.functional as F
import numpy as np
import re
from codes.utils.upload import save_data
from transformers import AutoTokenizer, AutoModelForMaskedLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
stopwords = ['/', ' ', ',', '', '.', '，', '#', '(', ')', '（', '）', ':', '-']


class BertEmbedding(object):
    def __init__(self, dataset_config):
        
        print('start bert model')
        self.get_cls = dataset_config.get_cls
        self.padding = dataset_config.padding
        self.bert_path = dataset_config.bert_path
        self.max_len = dataset_config.max_len
        self.load_bert()
        self.unk_idx = dataset_config.unk
        self.eos_idx = dataset_config.eos
        self.bos_idx = dataset_config.bos
        self.pad_idx = dataset_config.pad
        self.mask_idx =  dataset_config.mask

    def get_vocab_len(self):
        self.vocab_length = 30522
        return self.vocab_length

    def load_bert(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
        print('tokenizer fast', self.tokenizer.is_fast)
        self.model = AutoModelForMaskedLM.from_pretrained(self.bert_path, output_hidden_states=True)

    def tokenize(self, sentence, unk=False, get_cls=False):
        if self.padding:
            input_ids = torch.tensor(self.tokenizer.encode(sentence,
                                                           add_special_tokens=True,
                                                           padding='max_length',
                                                           truncation=True,
                                                           max_length=self.max_len))
        else:
            input_ids = torch.tensor(self.tokenizer.encode(sentence,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=self.max_len))
        # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
        # input_ids = input_ids.to(device)
        padding_mask = torch.zeros(input_ids.shape)
        padding_mask[input_ids != self.pad_idx] = 1
        return input_ids, padding_mask

    def tokenizer_sentences(self, sentences: list, unk=False, get_cls=False):
        if self.padding:
            result = self.tokenizer(sentences,
                                       add_special_tokens=True,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len)
        else:
            result = self.tokenizer(sentences,
                                       add_special_tokens=True,
                                       truncation=True,
                                       max_length=self.max_len)
        # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
        # input_ids = input_ids.to(device)
        input_ids = result['input_ids']
        padding_mask = result['attention_mask']
        return input_ids, padding_mask


    def emb_from_idx(self, inputs, get_cls=True):
        self.model.to(device)
        self.model.eval()
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        # 进行编码
        with torch.no_grad():
            outputs = self.model(inputs)
            if self.get_cls:
                # get the CLS value in the last layer
                hidden_states = outputs['hidden_states']
                output = hidden_states[-1][0, 0, :]
            else:
                # get the hidden state value in the last layer
                hidden_states = outputs['hidden_states']
                output = hidden_states[-1].squeeze(0)
                # output = torch.cat((hidden_states[-1].squeeze(0), hidden_states[-2].squeeze(0)), dim=1)
        return output

    def emb(self, sentence, get_cls=True, unk=False):
        input_ids, padding_mask = self.tokenize(sentence)
        input_ids = input_ids.to(device)
        padding_mask = input_ids[input_ids == 0]

        self.get_cls = get_cls
        # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
        self.model.to(device)
        self.model.eval()
        # 进行编码
        output = self.emb_from_idx(input_ids, get_cls=get_cls)
        return output.cpu(), padding_mask

    def get_sentence_by_index(self, index_list):
        pad_mask = []
        true_idx = []
        for index in index_list:
            index = index
            true_idx.append(index)
            pad_mask.append(1)
            if index == self.eos_idx:
                break
        sentence = self.tokenizer.convert_ids_to_tokens(true_idx)
        while len(true_idx) < self.max_len:
            true_idx.append(self.pad_idx)
            pad_mask.append(0)

        if sentence[0] == '[CLS]':
            sentence = sentence[1:]
        if sentence[-1] == '[SEP]':
            sentence = sentence[:-1]
        sentence = ' '.join(sentence)
        return sentence, true_idx, pad_mask

    def dataset_emb(self, text_list):
        print('start get embedding')
        total_result = []
        for sentence in text_list:
            result = self.emb(sentence)
            total_result.append(result)
        return total_result

    def get_bos(self):
        return None

    def get_eos(self):
        return None


class W2VEmbedding(object):
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.padding = dataset_config.padding
        self.max_len = dataset_config.max_len

        self.load_emb()

        self.unk_idx = self.vocab_dict['unk']
        self.eos_idx = self.vocab_dict['eos']
        self.bos_idx = self.vocab_dict['bos']
        self.pad_idx = self.vocab_dict['pad']
        self.mask_idx = self.vocab_dict['mask']
        # get eos
        self.eos = self.get_eos()
        self.pad_vec = self.embedding_layer(torch.tensor([self.pad_idx]))

    def load_emb(self):
        # 词典
        self.vocab_dict = np.load(self.config.w2v_vocab_path, allow_pickle=True).item()
        # print('vocab dict', self.vocab_dict.keys())
        # 用作反向找单词
        self.index_vocab_dict = dict(zip(self.vocab_dict.values(), self.vocab_dict.keys()))
        print('vocab length is', len(self.index_vocab_dict))
        # embeddeing layer weight
        weight = torch.load(self.config.w2v_model_path)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(weight, freeze=True)

    def get_vocab_len(self):
        self.vocab_length = len(self.index_vocab_dict)
        return self.vocab_length

    def tokenize(self, sentence, get_cls=False, unk=False):
        # print('before cut sentence', sentence)
        words = sentence.lower().split(' ')
        words = [x for x in words if x not in stopwords]
        words = [x for x in words if x.isdigit() is False]
        # print('after cut sentence', words)
        input_list = []
        padding_mask = []
        count = 0
        if get_cls is False:
            input_list.append(self.bos_idx)
            padding_mask.append(1)
        else:
            pass

        for word in words:
            word = word.lower()
            # print('word is', word)
            if len(input_list) >= self.max_len - 1:
                break
            if word in self.vocab_dict.keys():
                input_list.append(self.vocab_dict[word])
                padding_mask.append(1)
                count += 1
            else:
                if unk:
                    input_list.append(self.unk_idx)
                    padding_mask.append(1)
                else:
                    pass
        if len(input_list) >= self.max_len - 1:
            input_list = input_list[:self.max_len-1]
            padding_mask = padding_mask[:self.max_len-1]

        if get_cls is False:
            input_list.append(self.eos_idx)
            padding_mask.append(1)
        # pad to max len
        if self.padding and (get_cls is False):
            pad_mask_first_index = 0  # 第一个pad要加入考虑
            while len(input_list) < self.max_len:
                input_list.append(self.pad_idx)
                padding_mask.append(pad_mask_first_index)
                pad_mask_first_index = 0
        #print('after mapping sentence', input_list)
        
        assert len(input_list) == len(padding_mask)
        assert count != 0, 'No word in this sentence can be find by w2v: %s' % sentence
        return torch.tensor(input_list), torch.tensor(padding_mask).float()

    def emb_from_idx(self, input_idx):
        save_device = input_idx.device
        input_idx = input_idx.to(self.embedding_layer.weight.device)
        vector = self.embedding_layer(input_idx)
        return vector.float().detach().to(save_device)

    def emb(self, sentence, get_cls=False, unk=False):
        input_list, padding_mask = self.tokenize(sentence, get_cls=get_cls, unk=unk)
        vector = self.emb_from_idx(input_list)
        if get_cls:
            # vector = torch.mean(vector, dim=0)
            vector = torch.max(vector, dim=0)[0]
            vector = vector.unsqueeze(0)
        return vector.float().detach(), padding_mask

    def find_sim_word(self, one_sentence_output):
        # vector: [max_len, vector]
        sim = F.cosine_similarity(one_sentence_output.cpu().unsqueeze(1),
                                  self.embedding_layer.weight.unsqueeze(0),
                                  dim=2)
        index_list = torch.argmax(sim, dim=1).tolist()
        sentence, true_idx, pad_mask = self.get_sentence_by_index(index_list)
        return sentence, true_idx, pad_mask

    def dataset_emb(self, text_list):
        print('start get embedding')
        total_result = []
        for sentence in text_list:
            result = self.emb(sentence)
            total_result.append(result)
        return total_result

    def get_bos(self):
        bos_vec = self.embedding_layer(torch.tensor([self.bos_idx]))
        return bos_vec.detach()

    def get_eos(self):
        eos_vec = self.embedding_layer(torch.tensor([self.eos_idx]))
        return eos_vec.detach()

    def cut_and_pad(self, sentence_output):
        """
        找到和eos最近的一个词作为结束，后面设置为pad
        """
        sim = F.cosine_similarity(self.eos.unsqueeze(1).to(sentence_output.device),
                                  sentence_output,
                                  dim=2)
        index_list = torch.argmax(sim, dim=1).cpu().tolist()
        result = []
        for i in range(len(index_list)):
            cuted_sentence = sentence_output[i, :index_list[i], :].to(sentence_output.device)
            pad_length = self.max_len - cuted_sentence.shape[0]
            pad_mat = self.pad_vec.repeat([pad_length, 1]).to(sentence_output.device)
            paded_sentence = torch.cat([cuted_sentence, pad_mat], dim=0).unsqueeze(0)
            result.append(paded_sentence)
        result = torch.cat(result, dim=0)
        return result.detach()

    def get_sentence_by_index(self, index_list):
        sentence = []
        pad_mask = []
        true_idx = []
        for index in index_list:
            index = index
            word = self.index_vocab_dict[index]
            sentence.append(word)
            true_idx.append(index)
            pad_mask.append(1)
            if index == self.eos_idx:
                break
        sentence = ' '.join(sentence)
        while len(true_idx) < self.max_len:
            true_idx.append(self.pad_idx)
            pad_mask.append(0)
        return sentence, true_idx, pad_mask

    def cut_and_pad_idx(self, sentence_output):
        # vector: [max_len, vector]
        print(self.eos.shape, sentence_output.shape)
        sim = F.cosine_similarity(self.eos.unsqueeze(1).to(sentence_output.device),
                                  sentence_output,
                                  dim=2)
        index_list = torch.argmax(sim, dim=1).cpu().tolist()
        result = []
        for i in range(len(index_list)):
            cuted_sentence = sentence_output[i, :index_list[i], :].to(sentence_output.device)
            pad_length = self.max_len - cuted_sentence.shape[0]
            pad_mat = self.pad_vec.repeat([pad_length, 1]).to(sentence_output.device)
            paded_sentence = torch.cat([cuted_sentence, pad_mat], dim=0).unsqueeze(0)
            result.append(paded_sentence)
        result = torch.cat(result, dim=0)
        return result.detach()


class EmbeddingLayer(object):
    '''
    emb_layer = EmbeddingLayer(config)
    emb_layer.data_emb(save_file_name, text_list, get_cls, padding, save_oss)
    '''
    def __init__(self, dataset_config):
        self.emb_type = dataset_config.emb_type

        if self.emb_type == 'bert':
            self.emb_model = BertEmbedding(dataset_config)
        elif self.emb_type == 'w2v':
            self.emb_model = W2VEmbedding(dataset_config)

    def data_emb(self, save_file_name, text_list, get_cls=True, padding=True, save_oss=True):
        total_result_list = self.emb_model.dataset_emb(text_list, get_cls=get_cls, padding=padding)
        self.save_emb_result(save_file_name, total_result_list, save_oss=save_oss)

    def emb_one_text(self, text, get_cls=False, unk=False):
        result, padding_maks = self.emb_model.emb(text, get_cls=get_cls, unk=unk)
        return result, padding_maks

    def save_emb_result(self, save_file_name, total_result, save_oss=True):
        # save files
        local_path = os.path.join(self.save_path, save_file_name)
        oss_path = os.path.join(self.save_oss_path, save_file_name)
        torch.save(total_result, local_path)
        # 是否上传oss
        if save_oss:
            save_data(local_path, oss_path)

    def get_bos(self):
        bos_vec = self.emb_model.get_bos()
        return bos_vec

    def get_eos(self):
        eos_vec = self.emb_model.get_eos()
        return eos_vec
