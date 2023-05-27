import os
import torch
import numpy as np
import jieba
import string
from codes.utils.upload import save_data
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.models.bart.modeling_bart import BartEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
stopwords = ['are', "should've", 'some', 'for', 'being', 'few', 'then', 'them', 're', 'no', "you're", 'am', "wouldn't", 'this', 'in', 'you', 'isn', 'off', 'couldn', 'myself', 'any', 'those', 'first', 'he', 'with', 'between', 'discharge', 'shan', 'has', 'do', 'on', 'chief', 'so', 'ain', "couldn't", "she's", 'patient', 'of', 've', "needn't", 'don', 'that', 'mightn', 'date', 'why', "don't", 'these', 'into', 'having', 'does', 'her', 'yours', 'sex', "mightn't", 'yourself', "shouldn't", 'down', 'out', 'own', 'admission', 'm', "weren't", 'did', 'which', 'wouldn', 'my', 'its', 'our', 'herself', 'ma', 'mustn', 'were', 'name', "won't", "hadn't", 'or', 'same', "hasn't", 'should', 'from', 'just', 'd', 'past', 'such', 'above', 'too', 'i', 'only', 'won', 'yourselves', 'o', 'before', 'haven', 'hers', 'after', 's', 'course', 'doesn', 'while', 'been', 'by', 'once', 'your', 'during', 'didn', 'most', "you'll", 'below', 'last', 'where', 'wasn', 'over', 'up', 'she', 'under', 'and', 'him', 'who', 'until', 'one', 'as', 'both', "haven't", 'through', 'had', 'each', "that'll", 'more', "didn't", 'there', 'nor', 'his', 'shouldn', 'service', "you've", 'a', 'they', 'it', 'other', "doesn't", 'y', 'complaint', 'will', 'll', 'hasn', "it's", 'be', 'further', 'now', 'we', "mustn't", 'weren', 'their', 't', 'himself', 'day', "shan't", "wasn't", 'because', 'when', 'itself', 'all', 'can', 'than', 'at', 'needn', 'the', 'again', 'family', 'against', 'hospital', 'ourselves', 'what', "aren't", 'was', 'very', 'themselves', 'about', 'not', 'hadn', "isn't", 'me', 'but', 'to', 'here', 'if', 'history', 'whom', 'ours', 'doing', 'have', 'aren', 'an', 'is', 'theirs', 'how', 'birth', "you'd"]


class BertEmbedding(object):
    def __init__(self, dataset_config):
        print('start bert model')
        self.get_cls = dataset_config.get_cls
        self.padding = dataset_config.padding
        self.bert_path = dataset_config.bert_path
        self.max_len = dataset_config.max_len
        self.load_bert()

    def load_bert(self):
        # self.tokenizer = AutoTokenizer.from_pretrained(self.bart_path)
        # self.model = BartEncoder.from_pretrained(self.bart_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.model = AutoModelForMaskedLM.from_pretrained(self.bert_path, output_hidden_states=True)

    def emb(self, sentence, get_cls=False):
        self.get_cls = get_cls
        if self.padding:
            input_ids = torch.tensor([self.tokenizer.encode(sentence,
                                                            add_special_tokens=True,
                                                            padding='max_length',
                                                            truncation=True,
                                                            max_length=self.max_len)])
        else:
            input_ids = torch.tensor([self.tokenizer.encode(sentence,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=self.max_len)])
        self.model.to(device)
        self.model.eval()
        # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
        input_ids = input_ids.to(device)
        padding_mask = input_ids[input_ids == 0]

        # 进行编码
        with torch.no_grad():
            outputs = self.model(input_ids)
            if self.get_cls:
                # get the CLS value in the last layer
                hidden_states = outputs['hidden_states']
                output = hidden_states[-1][0, 0, :]
            else:
                # get the hidden state value in the last layer
                hidden_states = outputs['hidden_states']
                output = hidden_states[-1].squeeze(0)
                # output = torch.cat((hidden_states[-1].squeeze(0), hidden_states[-2].squeeze(0)), dim=1)
        return output.cpu(), padding_mask

    def dataset_emb(self, text_list):
        print('start get embedding')
        total_result = []
        for sentence in text_list:
            result = self.emb(sentence)
            total_result.append(result)
        return total_result

    def get_bos(self):
        return None


class W2VEmbedding(object):
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.padding = dataset_config.padding
        self.max_len = dataset_config.max_len
        self.unk_idx = dataset_config.unk
        self.eos_idx = dataset_config.eos
        self.bos_idx = dataset_config.bos
        self.pad_idx = dataset_config.pad
        self.load_emb()

    def load_emb(self):
        self.vocab_dict = np.load(self.config.w2v_vocab_path, allow_pickle=True).item()
        weight = torch.load(self.config.w2v_model_path)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(weight)

    def tokenize(self, sentence, get_cls=False):
        punct = string.punctuation.replace('-', '') + ''.join(["``", "`", "..."])
        trantab = str.maketrans(punct, len(punct) * ' ')
        sentence = sentence.lower().translate(trantab)
        words = sentence.strip().split()
        words = [x for x in words if x not in stopwords]
        input_list = []
        padding_mask = []
        if get_cls is False:
            input_list.append(self.bos_idx)
            padding_mask.append(1)
        else:
            pass

        for word in words:
            if len(input_list) < self.max_len - 1:
                pass
            else:
                break
            if word in self.vocab_dict.keys():
                input_list.append(self.vocab_dict[word])
                padding_mask.append(1)
            else:
                input_list.append(self.unk_idx)
                padding_mask.append(1)
        if get_cls is False:
            input_list.append(self.eos_idx)
            padding_mask.append(1)
        # pad to max len
        if self.padding and (get_cls is False):
            while len(input_list) < self.max_len:
                input_list.append(self.pad_idx)
                padding_mask.append(0)
        assert len(input_list) == len(padding_mask)
        return torch.tensor(input_list), torch.tensor(padding_mask).float()

    def emb(self, sentence, get_cls=False):
        input_list, padding_mask = self.tokenize(sentence, get_cls=get_cls)
        vector = self.embedding_layer(input_list)
        if get_cls:
            vector = torch.mean(vector, dim=0)
            vector = vector.unsqueeze(0)
        return vector.float().detach(), padding_mask

    def dataset_emb(self, text_list):
        print('start get embedding')
        total_result = []
        for sentence in text_list:
            result = self.emb(sentence)
            total_result.append(result)
        return total_result

    def get_bos(self):
        bos_vec = self.embedding_layer(torch.tensor([self.bos_idx]))
        return bos_vec


class EmbeddingLayer(object):
    '''
    emb_layer = EmbeddingLayer(config)
    emb_layer.data_emb(save_file_name, text_list, get_cls, padding, save_oss)
    '''
    def __init__(self, dataset_config):
        self.emb_type = dataset_config.emb_type
        # self.save_path = config.save_path
        # self.save_oss_path = config.save_oss_path
        # self.upload_oss = config.upload_oss

        if self.emb_type == 'bert':
            self.emb_model = BertEmbedding(dataset_config)
        elif self.emb_type == 'w2v':
            self.emb_model = W2VEmbedding(dataset_config)

    def data_emb(self, save_file_name, text_list, get_cls=True, padding=True, save_oss=True):
        total_result_list = self.emb_model.dataset_emb(text_list, get_cls=get_cls, padding=padding)
        self.save_emb_result(save_file_name, total_result_list, save_oss=save_oss)

    def emb_one_text(self, text, get_cls=False):
        result, padding_maks = self.emb_model.emb(text, get_cls=get_cls)
        return result, padding_maks

    def save_emb_result(self, save_file_name, total_result, save_oss=True):
        # save files
        local_path = os.join(self.save_path, save_file_name)
        oss_path = os.join(self.save_oss_path, save_file_name)
        torch.save(total_result, local_path)
        # 是否上传oss
        if save_oss:
            save_data(local_path, oss_path)

    def get_bos(self):
        bos_vec = self.emb_model.get_bos()
        return bos_vec
