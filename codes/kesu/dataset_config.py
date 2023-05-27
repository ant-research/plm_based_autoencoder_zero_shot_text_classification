import os


class KESUDataConfig():
    def __init__(self, easy_label=False, train_type = 'v0'):
        self.task_type = 'Single_Label'
        # path = 'kesu_easy'
        path = 'kesu_zero-shot'
        # dataset download and path config
        if easy_label:
            self.dataset_oss_data_path = f'cunyin/{path}'
            self.dataset_oss_emb_path = f'cunyin/{path}/emb_output'
        else:
            self.dataset_oss_data_path = f'cunyin/{path}'
            self.dataset_oss_emb_path = f'cunyin/{path}/emb_output'
        self.dataset_local_data_path = './data/'
        # unlabeled data
        self.unlabel_oss_path = os.path.join(self.dataset_oss_data_path, 'unlabeled_data.csv')
        self.unlabel_local_path = os.path.join(self.dataset_local_data_path, 'unlabeled.csv')
        # label description
        self.label_oss_path = os.path.join(self.dataset_oss_data_path, 'label.csv')
        self.label_local_path = os.path.join(self.dataset_local_data_path, 'label.csv')
        self.label_description = False
        # labeled dataset
        self.oss_path = {}
        self.local_path = {}
        self.dataset_oss_emb_path = f'cunyin/{path}/emb_output'
        for data_type in ['train', 'eval', 'test']:
            if data_type == 'train':
                self.oss_path[data_type] = os.path.join(self.dataset_oss_data_path, '%s.csv' % data_type)
                self.local_path[data_type] = os.path.join(self.dataset_local_data_path, '%s.csv' % data_type)
                self.oss_path[data_type + '_zip'] = os.path.join(self.dataset_oss_emb_path, '%s.zip' % data_type)
                self.local_path[data_type + '_zip'] = os.path.join(self.dataset_local_data_path, '%s.zip' % data_type)
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


class KESUDataConfig_W2V(KESUDataConfig):
    def __init__(self, emb_type='bert', easy_label=False):
        super().__init__(easy_label=easy_label)
        self.task_type = 'Single_Label'

        # w2v download path
        self.w2v_model_path = './pretrained_model/w2v_embedding.pt'
        self.w2v_vocab_path = './pretrained_model/vocab.npy'
        self.w2v_oss_path = 'cunyin/pretrained_model/w2v_yuque'
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
        self.max_len = 30
        self.emb_size = 200
        self.unk = 73725
        self.pad = 56176
        self.eos = 68531
        self.bos = 73303
        self.mask = 0


class KESUDataConfig_BERT_T(KESUDataConfig):
    def __init__(self, easy_label=False, emb_type='bert'):
        super().__init__(easy_label=easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './pretrained_model/bert_t'  # bert save path
        self.bert_oss_path = 'cunyin/pretrained_model/kesu_bert_pretrained/'  # remote oss bert path
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


class KESUDataConfig_BERT_P(KESUDataConfig):
    def __init__(self, emb_type='bert', easy_label=False):
        super().__init__(easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.bert_path = './pretrained_model/bert_p'  # bert save path
        self.bert_oss_path = 'cunyin/pretrained_model/kesu_bert_pretrained'  # remote oss bert path
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


class KESUDataConfig_GPT(KESUDataConfig):
    def __init__(self, emb_type='gpt', easy_label=False):
        super().__init__(easy_label)
        self.task_type = 'Single_Label'
        # embedding config
        self.gpt2_path = './pretrained_model/gpt2'  # bert save path
        self.gpt2_oss_path = 'cunyin/pretrained_model/kesu_gpt_pretrained'  # remote oss bert path
        self.gpt2_file_list = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "training_args.bin"]

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


class KESUDataConfig_Local(KESUDataConfig):
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


KESUDataConfigDICT = {
    'w2v': KESUDataConfig_W2V,
    'bert_t': KESUDataConfig_BERT_T,
    'bert_p': KESUDataConfig_BERT_P,
    'gpt2': KESUDataConfig_GPT,
}
