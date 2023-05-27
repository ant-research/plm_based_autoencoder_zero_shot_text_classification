import os


class SituationDataConfig():
    def __init__(self, easy_label=False, train_type = 'v0'):
        print(f"train type is {train_type}")
        self.task_type = 'Single_Label'

        # dataset download and path config
        if easy_label:
            self.dataset_oss_data_path = 'situation'
            self.dataset_oss_emb_path = 'situation/emb_output'
        else:
            self.dataset_oss_data_path = 'situation'
            self.dataset_oss_emb_path = 'situation/emb_output'
        self.dataset_local_data_path = './data/situation'
        # unlabeled data
        self.unlabel_oss_path = os.path.join(self.dataset_oss_data_path, 'unlabeled.csv')
        self.unlabel_local_path = os.path.join(self.dataset_local_data_path, 'unlabeled.csv')
        # label description
        self.label_oss_path = os.path.join(self.dataset_oss_data_path, 'label.csv')
        self.label_local_path = os.path.join(self.dataset_local_data_path, 'label.csv')
        self.label_description = False
        # labeled dataset
        self.oss_path = {}
        self.local_path = {}
        self.dataset_oss_emb_path = 'situation/emb_output'
        for data_type in ['train', 'eval', 'test']:
            if data_type == 'train':
                if train_type == "v0":
                    self.oss_path[data_type] = os.path.join(self.dataset_oss_data_path, '%sv0.csv' % data_type)
                    self.local_path[data_type] = os.path.join(self.dataset_local_data_path, '%sv0.csv' % data_type)
                    self.oss_path[data_type + '_zip'] = os.path.join(self.dataset_oss_emb_path, '%sv0.zip' % data_type)
                    self.local_path[data_type + '_zip'] = os.path.join(self.dataset_local_data_path, '%sv0.zip' % data_type)
                elif train_type == "v1":
                    self.oss_path[data_type] = os.path.join(self.dataset_oss_data_path, '%sv1.csv' % data_type)
                    self.local_path[data_type] = os.path.join(self.dataset_local_data_path, '%sv1.csv' % data_type)
                    self.oss_path[data_type + '_zip'] = os.path.join(self.dataset_oss_emb_path, '%sv1.zip' % data_type)
                    self.local_path[data_type + '_zip'] = os.path.join(self.dataset_local_data_path, '%sv1.zip' % data_type)
                else:
                    raise KeyError
                self.train_npy_length = 1
                data_length = 50000
                for i in range(0, self.train_npy_length):
                    number = i * data_length
                    self.local_path[data_type + '_' + str(number)] = os.path.join(self.dataset_local_data_path, 'train_%d.npy' % number)
                    self.local_path[data_type + '_da_' + str(number)] = os.path.join(self.dataset_local_data_path, 'train_da_%d.npy' % number)
            else:
                self.oss_path[data_type] = os.path.join(self.dataset_oss_data_path, '%s.csv' % data_type)
                self.local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s.csv' % data_type)
                self.oss_path[data_type + '_zip'] = os.path.join(self.dataset_oss_emb_path, '%s.zip' % data_type)
                self.local_path[data_type + '_zip'] = os.path.join(self.dataset_local_data_path, '%s.zip' % data_type)  
                if data_type == 'test':
                    self.test_npy_length = 1
                    data_length = 50000
                    for i in range(0, self.test_npy_length):
                        number = i * data_length

                        self.local_path[data_type + '_' + str(number)] = os.path.join(self.dataset_local_data_path, 'test_%d.npy' % number)
                        self.local_path[data_type + '_da_' + str(number)] = os.path.join(self.dataset_local_data_path, 'test_da_%d.npy' % number)                
                if data_type == 'eval':
                    self.eval_npy_length = 1 
                    data_length = 50000
                    for i in range(0, self.eval_npy_length):
                        number = i * data_length
                        self.local_path[data_type + '_' + str(number)] = os.path.join(self.dataset_local_data_path, 'eval_%d.npy' % number)
                        self.local_path[data_type + '_da_' + str(number)] = os.path.join(self.dataset_local_data_path, 'eval_da_%d.npy' % number)        
                      
        self.emb_type = None

    def embed_config(self):
        '''
        embedding config
        '''
        assert self.emb_type is not None
        self.save_local_path = {}
        self.save_oss_path = {}
        self.save_da_local_path = {}
        self.save_da_oss_path = {}
        self.save_label_local_path = os.path.join(self.dataset_local_data_path, 'label_%s.pt' % self.emb_type)
        self.save_label_oss_path = os.path.join(self.dataset_oss_emb_path, 'label_%s.pt' % self.emb_type)
        self.save_easy_label_local_path = os.path.join(self.dataset_local_data_path, 'easy_label_%s.pt' % self.emb_type)
        self.save_easy_label_oss_path = os.path.join(self.dataset_oss_emb_path, 'easy_label_%s.pt' % self.emb_type)
        for data_type in ['train', 'eval', 'test']:
            self.save_local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s_%s.pt' %
                                                           (data_type, self.emb_type))
            self.save_da_local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s_%s_da.pt' %
                                                              (data_type, self.emb_type))
            self.save_oss_path[data_type] = os.path.join(self.dataset_oss_emb_path, '%s_%s.pt' %
                                                         (data_type, self.emb_type))
            self.save_da_oss_path[data_type] = os.path.join(self.dataset_oss_emb_path, '%s_%s_da.pt' %
                                                            (data_type, self.emb_type))

    def da(self, da_type='eda'):
        self.da_type = da_type


class SituationDataConfig_W2V(SituationDataConfig):
    def __init__(self, emb_type='bert', easy_label=False):
        super().__init__(easy_label=easy_label)
        self.task_type = 'Single_Label'

        # w2v download path
        self.w2v_model_path = './pretrained_model/w2v_embedding.pt'
        self.w2v_vocab_path = './pretrained_model/vocab.npy'
        self.w2v_oss_path = 'pretrained_model/w2v_wos'
        self.w2v_local_path = './pretrained_model'
        self.w2v_file_list = ['w2v_embedding.pt', 'vocab.npy']

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self):
        '''
        embedding config
        '''
        print('use w2v type')
        self.emb_type = 'w2v'
        super().embed_config()
        self.padding = True
        self.max_len = 50
        self.emb_size = 200
        self.unk = 201534
        self.pad = 10109
        self.eos = 28069
        self.bos = 26828
        self.mask = 0


class SituationDataConfig_BERT_T(SituationDataConfig):
    def __init__(self, easy_label=False, emb_type='bert'):
        super().__init__(easy_label=easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './pretrained_model/bert_t'  # bert save path
        self.bert_oss_path = 'pretrained_model/sit_bert/'  # remote oss bert path
        self.bert_file_list = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json',
                               'tokenizer.json', 'special_tokens_map.json']

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self):
        '''
        embedding config
        '''
        print('use bert type')
        self.emb_type = 'bert'
        super().embed_config()
        self.get_cls = True
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.vocab_len = 30522
        self.pad = 0
        self.eos = 102
        self.bos = 101
        self.unk = 100
        self.mask = 103


class SituationDataConfig_BERT_P(SituationDataConfig):
    def __init__(self, emb_type='bert', easy_label=False):
        super().__init__(easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './pretrained_model/bert_p'  # bert save path
        self.bert_oss_path = 'pretrained_model/sit_bert'  # remote oss bert path
        self.bert_file_list = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json',
                               'tokenizer.json', 'special_tokens_map.json']

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self, emb_type='bert'):
        '''
        embedding config
        '''
        print('use bert type')
        self.emb_type = 'bert'
        super().embed_config()
        self.get_cls = False
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.vocab_len = 30522
        self.pad = 0
        self.eos = 102
        self.bos = 101
        self.unk = 100
        self.mask = 103


class SituationDataConfig_GPT(SituationDataConfig):
    def __init__(self, emb_type='gpt', easy_label=False):
        super().__init__(easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.gpt2_path = './pretrained_model/gpt2'  # bert save path
        self.gpt2_oss_path = 'pretrained_model/sit_gpt'  # remote oss bert path
        self.gpt2_file_list = ["added_tokens.json", "config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "training_args.bin"]

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self, emb_type='gpt'):
        '''
        embedding config
        '''
        print('use gpt type')
        self.emb_type = 'gpt'
        super().embed_config()
        self.get_cls = False
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.unk_idx = 50256
        self.pad_idx = 50257
        self.eos_idx = 50259
        self.bos_idx = 50258
        self.mask_idx = 50260


class SituationDataConfig_BERT_T_LOCAL(SituationDataConfig):
    def __init__(self, easy_label=False, emb_type='bert', train_type="v0"):
        super().__init__(easy_label=easy_label, train_type=train_type)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './models/sit_bert'  # bert save path
        self.bart_path = './models/BART_Encoder'
        self.bert_oss_path = 'pretrained_model/sit_bert/'  # remote oss bert path
        self.bert_file_list = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json',
                               'tokenizer.json', 'special_tokens_map.json']

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self):
        '''
        embedding config
        '''
        print('use bert type')
        self.emb_type = 'bert'
        super().embed_config()
        self.get_cls = True
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.vocab_len = 30522
        self.pad = 0
        self.eos = 102
        self.bos = 101
        self.unk = 100
        self.mask = 103


class SituationDataConfig_BERT_P_LOCAL(SituationDataConfig):
    def __init__(self, emb_type='bert', easy_label=False, train_type="v0"):
        super().__init__(easy_label=easy_label, train_type=train_type)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './models/sit_bert'  # bert save path
        self.bart_path = './models/BART_Encoder'
        self.bert_oss_path = 'pretrained_model/sit_bert'  # remote oss bert path
        self.bert_file_list = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json',
                               'tokenizer.json', 'special_tokens_map.json']

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self, emb_type='bert'):
        '''
        embedding config
        '''
        print('use bert type')
        self.emb_type = 'bert'
        super().embed_config()
        self.get_cls = False
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.vocab_len = 30522
        self.pad = 0
        self.eos = 102
        self.bos = 101
        self.unk = 100
        self.mask = 103


class SituationDataConfig_GPT_LOCAL(SituationDataConfig):
    def __init__(self, emb_type='gpt', easy_label=False, train_type="v0"):
        super().__init__(easy_label=easy_label, train_type=train_type)
        self.task_type = 'Single_Label'
        # embedding config
        self.gpt2_path = './models/sit_gpt'  # bert save path
        self.bart_path = './models/BART_Decoder'
        self.gpt2_oss_path = 'pretrained_model/sit_gpt'  # remote oss bert path
        self.gpt2_file_list = ["added_tokens.json", "config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "training_args.bin"]

        # embedding setting
        self.embed_config()
        # augmentation setting
        self.da()

    def embed_config(self, emb_type='gpt'):
        '''
        embedding config
        '''
        print('use gpt type')
        self.emb_type = 'gpt'
        super().embed_config()
        self.get_cls = False
        self.padding = True
        self.max_len = 50
        self.emb_size = 768
        self.unk_idx = 50256
        self.pad_idx = 50257
        self.eos_idx = 50259
        self.bos_idx = 50258
        self.mask_idx = 50260


class SituationDataConfig_Local(SituationDataConfig):
    def __init__(self, emb_type='bert'):
        super().__init__()
        # dataset download and path config
        self.dataset_local_data_path = '/data/'
        self.label_local_path = os.path.join(self.dataset_local_data_path, 'clean_label.csv')
        self.local_path = {}
        for data_type in ['train', 'eval', 'test']:
            self.local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s.csv' % data_type)
        # embedding setting
        self.embed_config(emb_type=emb_type)
        # augmentation setting
        self.da()
        raise NotImplementedError

    def embed_config(self, emb_type='bert'):
        '''
        embedding config
        '''
        if emb_type == 'bert':
            print('use bert type')
            self.emb_type = 'bert'
            self.get_cls = False
            self.padding = True
            self.max_len = 30
            self.emb_size = 768
        elif emb_type == 'w2v':
            print('use w2v type')
            self.emb_type = 'w2v'
            self.padding = True
            self.max_len = 30
            self.emb_size = 200
        else:
            raise KeyError
        self.save_local_path = {}
        self.save_da_local_path = {}
        self.save_label_local_path = os.path.join(self.dataset_local_data_path, 'label_%s.pt' % emb_type)
        for data_type in ['train', 'eval', 'test']:
            self.save_local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s_%s.pt' %
                                                           (data_type, self.emb_type))
            self.save_da_local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s_%s_da.pt' %
                                                              (data_type, self.emb_type))

    def da(self, da_type='eda'):
        self.da_type = da_type


SituationDataConfigDICT = {
    'w2v': SituationDataConfig_W2V,
    'bert_t': SituationDataConfig_BERT_T_LOCAL,
    'bert_p': SituationDataConfig_BERT_P_LOCAL,
    'gpt2': SituationDataConfig_GPT_LOCAL,
}
