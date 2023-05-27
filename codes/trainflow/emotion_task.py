import os
import time
import torch
import zipfile
from codes.utils.requirement import install_synonyms, install_wordnet, install_owms
from codes.trainflow.taskflow import TaskFlow
from codes.utils import download_file, Logger
from codes.emotion.dataloader import (
    Emotion_Dataset,
    Emotion_Idx_Dataset,
    Emotion_Generated_Dataset,
    Emotion_Unlabeled_Dataset,
    Emotion_Idx_Unlabeled_Dataset,
    Emotion_Idx_Label_dataset,
    Emotion_Idx_Entailment_Dataset,
    Emotion_Generated_Enailment_Dataset
    )
from codes.emotion.emb import EmbeddingLayer
# import model config
from codes.emotion.model_config import ModelConfig
from codes.emotion.dataset_config import EmotionDataConfigDICT, EmotionDataConfig_Local
from model import ModelTrainer, ModelTrainerParallel
from model.one_trainer import OneTaskTrainer #, OneTaskTrainerBERT

class EmotionFlow(TaskFlow):
    def __init__(self, model_dict, dataset_dict, emb_type='w2v', parallel=False, local=True, model='New', debug=False,
                 init=False, download=True, last_label=True, pre_idx=True, decoder_gpt=False, entailment=False):
        '''
        model : ['New', 'TransICD']
        local: bool local train or aistudio train
        pre_idx: calculate idx before start train
        '''
        super().__init__()
        self.parallel = parallel
        self.emb_type = emb_type
        self.decoder_gpt = decoder_gpt
        self.init = init
        self.entailment = entailment
        print('##### init is ', init, 'entailment is', self.entailment)
        self.local = local
        self.download = download
        self.model = model
        self.debug = debug
        self.last_label = last_label
        self.pre_idx = pre_idx
        # dataset config
        if self.emb_type == 'bert':
            self.data_config = EmotionDataConfigDICT['bert_t'](emb_type=self.emb_type, **dataset_dict)
            self.data_bert_p_config = EmotionDataConfigDICT['bert_p'](emb_type=self.emb_type, **dataset_dict)
        elif self.emb_type == 'w2v':
            self.data_config = EmotionDataConfigDICT['w2v'](emb_type=self.emb_type, **dataset_dict)
        else:
            raise KeyError

        # choose the model config
        if model in ModelConfig.keys():
            self.model_config = ModelConfig[model](emb_size=self.data_config.emb_size,
                                                seq_len=self.data_config.max_len,
                                                **model_dict)
        else:
            self.model_config = ModelConfig['Other'](emb_size=self.data_config.emb_size,
                                                     seq_len=self.data_config.max_len,
                                                     **model_dict)
        self.model_config.task_type = self.data_config.task_type
        # push gpt config to model config
        if self.decoder_gpt == True:
            self.gpt_config = EmotionDataConfigDICT['gpt2'](emb_type=self.emb_type, **dataset_dict)
            self.model_config.gpt_config = self.gpt_config  # type: ignore
        self.logger = Logger().logger
        print(self.parallel, self.emb_type, self.model)

    def requirement_install(self):
        # if self.download:
        #     self.logger.info('start requirement')
        #     install_synonyms(local=self.local)  # for eda
        #     install_wordnet(local=self.local)  # for eda
        #     install_owms(local=self.local)
            
        if True:
            if self.emb_type == 'bert':
                self.logger.info('start download bert')
                # download bert_t
                if os.path.exists(self.data_config.bert_path) is False:  # type: ignore
                    os.mkdir(self.data_config.bert_path)  # type: ignore
                for file in self.data_config.bert_file_list:  # type: ignore
                    download_file(os.path.join(self.data_config.bert_oss_path, file),  # type: ignore
                                os.path.join(self.data_config.bert_path, file))  # type: ignore
                # download bert_p
                if os.path.exists(self.data_bert_p_config.bert_path) is False:
                    os.mkdir(self.data_bert_p_config.bert_path)
                for file in self.data_bert_p_config.bert_file_list:
                    download_file(os.path.join(self.data_bert_p_config.bert_oss_path, file),
                                os.path.join(self.data_bert_p_config.bert_path, file))
            elif self.emb_type == 'w2v':
                # download w2v file
                self.logger.info('start download word2vec file')
                for file_name in self.data_config.w2v_file_list:  # type: ignore
                    download_file(os.path.join(self.data_config.w2v_oss_path, file_name),  # type: ignore
                                  os.path.join(self.data_config.w2v_local_path, file_name))  # type: ignore
            if self.decoder_gpt == True:
                self.logger.info('start download gpt2')
                if os.path.exists(self.gpt_config.gpt2_path) is False:  # type: ignore
                    os.mkdir(self.gpt_config.gpt2_path)  # type: ignore
                for file in self.gpt_config.gpt2_file_list:
                    download_file(os.path.join(self.gpt_config.gpt2_oss_path, file),
                                os.path.join(self.gpt_config.gpt2_path, file))
            # download label data from oss
            self.logger.info('start download data')
            download_file(self.data_config.label_oss_path, self.data_config.label_local_path)
            # download total data from oss
            import zipfile
            for key in self.data_config.oss_path.keys():
                download_file(self.data_config.oss_path[key], self.data_config.local_path[key])
                if '_zip' in key:
                    print('unzip file %s' % self.data_config.local_path[key])
                    with zipfile.ZipFile(self.data_config.local_path[key], 'r') as zip_ref:
                        zip_ref.extractall(self.data_config.dataset_local_data_path)
            # for data_type in ['train', 'test', 'eval']:
            #     download_file(self.data_config.oss_path[data_type], self.data_config.local_path[data_type])
                
            # download unlabeled data from oss
            download_file(self.data_config.unlabel_oss_path, self.data_config.unlabel_local_path)
        else:
            pass
        # time.sleep(60)

    def preprocess_data(self):
        # download_bert
        init = self.init
        debug = self.debug
        local = self.local

        print('before embedding memory', torch.cuda.mem_get_info())
        # initial embedding layer
        if self.emb_type == 'w2v':
            self.emb_layer = EmbeddingLayer(self.data_config)  # embedding type
        elif self.emb_type == 'bert':
            self.emb_layer = EmbeddingLayer(self.data_config)  # embedding type
            self.emb_layer_p = EmbeddingLayer(self.data_bert_p_config)  # embedding type
        else:
            raise TypeError
        print('after embedding used memory', torch.cuda.mem_get_info())
        
        self.dataloader_list = []
        for data_type in ['train', 'test', 'eval']:
            self.logger.info('start data type %s' % data_type)
            # load dataset
            if data_type == 'train':
                label = True
            else:
                label = False
            if self.pre_idx:
                if self.entailment:
                    total_dataset = Emotion_Idx_Entailment_Dataset(self.data_config, data_type, logger=self.logger, label=label,
                                                            emb_layer=self.emb_layer, init=init, local=local, debug=debug)                    
                else:
                    total_dataset = Emotion_Idx_Dataset(self.data_config, data_type, logger=self.logger, label=label,
                                                            emb_layer=self.emb_layer, init=init, local=local, debug=debug)
            else:
                total_dataset = Emotion_Dataset(self.data_config, data_type, logger=self.logger, label=label,
                                                        emb_layer=self.emb_layer, init=init, local=local, debug=debug)
            self.dataloader_list.append(total_dataset)

        print('after dataset used memory', torch.cuda.mem_get_info())

        # load generate dataset
        self.logger.info('end unlabeled dataset build')
        self.label_mat = self.dataloader_list[0].label_mat
        self.model_config.class_num = self.label_mat.shape[0]  # type: ignore
        if self.entailment:
            self.generate_dataset = Emotion_Generated_Enailment_Dataset(class_num=self.model_config.class_num,
                                                                      dataset_config=self.data_config)
        else:
            self.generate_dataset = Emotion_Generated_Dataset(class_num=self.model_config.class_num,
                                                            dataset_config=self.data_config)
        # load unlabel dataset
        self.logger.info('end total dataset build')
        if self.pre_idx:
            self.unlabel_dataset = Emotion_Idx_Unlabeled_Dataset(dataset_config=self.data_config, init=init,
                                                             emb_layer=self.emb_layer, debug=debug, local=local)
            # self.unlabel_dataset = KESU_Unlabeled_Dataset(dataset_config=self.data_config, init=init,
            #                                               emb_layer=self.emb_layer, debug=debug, local=local)           
        else:
            self.unlabel_dataset = Emotion_Unlabeled_Dataset(dataset_config=self.data_config, init=init,
                                                          emb_layer=self.emb_layer, debug=debug, local=local)
        print('after unlabel and generate dataset used memory', torch.cuda.mem_get_info())
        self.dataloader_list.append(self.unlabel_dataset)
        
        # label description dataset
        label_dataset = Emotion_Idx_Label_dataset(self.data_config, data_type='eval', logger=self.logger, label=True,
                                            emb_layer=self.emb_layer, init=init, local=local, debug=debug)
        self.dataloader_list.append(label_dataset)
        
        
    def train(self):
        self.logger.info('start training, with label num %d' % self.model_config.class_num)
        #dataloader_list = [self.train_dataset, self.test_dataset, self.dev_dataset, self.unlabel_dataset]

        # build model trainer
        if self.parallel:
            model_trainer = ModelTrainerParallel[self.emb_type][self.model]
        else:
            model_trainer = ModelTrainer[self.emb_type][self.model]
        # build task trainer for cross validation training
        if self.emb_type == 'w2v':
            trainer = OneTaskTrainer(model_config=self.model_config,
                                    logger=self.logger,
                                    dataloader_list=self.dataloader_list,
                                    model_trainer=model_trainer,
                                    label_mat=self.label_mat,
                                    generated_dataset=self.generate_dataset,
                                    emb_layer_t=self.emb_layer,
                                    emb_layer_p=self.emb_layer,
                                    local=self.local)
        elif self.emb_type == 'bert':
            # 如果是bert类型，则两个encoder使用不一样的bert
            trainer = OneTaskTrainer(model_config=self.model_config,
                                    logger=self.logger,
                                    dataloader_list=self.dataloader_list,
                                    model_trainer=model_trainer,
                                    label_mat=self.label_mat,
                                    generated_dataset=self.generate_dataset,
                                    emb_layer_t=self.emb_layer,
                                    emb_layer_p=self.emb_layer_p,
                                    local=self.local)
        #     trainer = OneTaskTrainerBERT(model_config=self.model_config,
        #                                 logger=self.logger,
        #                                 dataloader_list=self.dataloader_list,
        #                                 model_trainer=model_trainer,
        #                                 label_mat=self.label_mat,
        #                                 generated_dataset=self.generate_dataset,
        #                                 emb_layer_t=self.emb_layer,
        #                                 emb_layer_p=self.emb_layer_p,
        #                                 local=self.local)
        else:
            raise TypeError

        trainer.train()
        # trainer.test()

    def test(self):
        pass
