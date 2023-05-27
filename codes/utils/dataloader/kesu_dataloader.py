import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from codes.utils import save_data, download_file



class KESU_Dataset(Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, data_type, emb_layer, logger, label=False, init=True, debug=False, local=True):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.logger = logger
        self.local = local  # local or mayi server
        self.dataset_config = dataset_config  # config
        
        self.emb_layer = emb_layer  # embedding layer
        # init data augumentation
        self.da_type = dataset_config.da_type  # data augmentation type
        from model.new_model.augmentation.eda_cn import EDA
        self.eda = EDA(num_aug=1)  # eda function
        # read label and dataset
        self.label_csv = pd.read_csv(dataset_config.label_local_path)
        df = pd.read_csv(dataset_config.local_path[data_type])
        # deal with label data
        y_data = list(df['label'])
        invalid_idx_list = self.deal_label(y_data)
        if label:
            if init:
                self.init_label()
            else:
                self.load_label()

        # deal with x data
        if init:
            self.x_text = list(df['data'])
            self.x_data = []
            self.x_da = []
            valid_y = []
            for idx in range(len(self.x_text)):
                # 去掉没有label的数据
                if idx in invalid_idx_list:
                    continue
                else:
                    if idx % 1000 == 0:
                        print(idx)
                    text = self.x_text[idx]
                    self.x_data.append(text)
                    # abletion study: da and contrastive loss
                    da_text = self.data_augmentation(text)
                    self.x_da.append(da_text)
                    valid_y.append(y_data[idx])
            new_df = pd.DataFrame()
            new_df['data'] = self.x_data
            new_df['da_data'] = self.x_da
            new_df['label'] = valid_y
            new_df.to_csv(dataset_config.local_path[data_type])
            save_data(dataset_config.oss_path[data_type], dataset_config.local_path[data_type])

        df = pd.read_csv(dataset_config.local_path[data_type])
        # debug: only use 2 batch size data to fast debug
        self.x_data = list(df['data'])
        self.x_da = list(df['da_data'])
        print(len(self.x_data), self.x_data)
        print(len(self.x_da), self.x_da)
        print(len(self.y_data), self.y_data)

    def deal_label(self, y_data):
        '''
        deal with label and get label matrix
        '''
        total_codes = list(self.label_csv['label'])
        self.logger.info('total code labels: %d' % len(total_codes))
        self.y_data = []
        seen_codes_list = []
        for label in y_data:
            index = total_codes.index(label)
            seen_codes_list.append(index)
            self.y_data.append(torch.tensor(index))
        self.logger.info('seen code labels: %d' % len(seen_codes_list))

    def init_label(self):
        '''
        build label matrix
        '''
        self.logger.info('init label')
        self.label_mat = []
        for text in self.label_csv['label']:
            label, pad = self.emb_layer.emb_one_text(text, get_cls=True)
            self.label_mat.append(label)
        self.label_mat = [label.unsqueeze(0) for label in self.label_mat]
        self.label_mat = torch.cat(self.label_mat, dim=0)
        print('label shape is', self.label_mat.shape)
        torch.save(self.label_mat, self.dataset_config.save_label_local_path)
        if self.local is False:
            save_data(self.dataset_config.save_label_oss_path, self.dataset_config.save_label_local_path)

    def load_label(self):
        '''
        load label data from oss
        '''
        if self.local is False:
            download_file(self.dataset_config.save_label_oss_path, self.dataset_config.save_label_local_path)
        self.logger.info('load label')
        self.label_mat = torch.load(self.dataset_config.save_label_local_path)

    def data_augmentation(self, text):
        eda_text = self.eda.eda(text)[0]
        return eda_text

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index], self.x_da[index]

    def __len__(self):
        return len(self.x_data)


class KESU_Generated_Dataset(Dataset):
    '''
    generated class and result
    '''
    def __init__(self, class_num):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.class_num = class_num
        self.x = []
        self.y = []
        for i in range(class_num):
            self.x.append(torch.tensor(i))
            self.y.append(torch.tensor(i))
        
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
