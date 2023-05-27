import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from codes.utils import save_data, download_file
from model.new_model.augmentation import eda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class KESU_Dataset(Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, data_type, emb_layer, logger, label=False, init=False, debug=False, local=True):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.logger = logger
        self.local = local  # local or mayi server
        self.dataset_config = dataset_config  # config
        self.label_description = dataset_config.label_description # 是否使用label的wikipedia描述作为embedding
        self.init = init
        self.label = label
        self.data_type = data_type
        self.dataset_config = dataset_config

        self.emb_layer = emb_layer  # embedding layer
        # init data augumentation
        self.da_type = dataset_config.da_type  # data augmentation type

        # read label and dataset
        self.label_csv = pd.read_csv(dataset_config.label_local_path)
        self.get_label_list()
        df = pd.read_csv(dataset_config.local_path[self.data_type])     

        # label
        y_data = self.deal_label(df)
        # x
        self.deal_x_data(df, y_data)

    def get_label_list(self):
        '''
        获取所有label的列表
        '''
        if self.label_description is True:
            self.label_list = self.label_csv['description']
        else:
            self.label_list = self.label_csv['label']

    def get_y_data(self, y_data):
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
        for text in self.label_list:
            try:
                label, pad = self.emb_layer.emb_one_text(text, get_cls=True)
            except Exception as e:
                print(e, text)
                label, pad = self.emb_layer.emb_one_text(text, get_cls=True, unk=True)
            self.label_mat.append(label)
        self.label_mat = [label.unsqueeze(0) for label in self.label_mat]
        self.label_mat = torch.cat(self.label_mat, dim=0).to(device)
        print('label shape is', self.label_mat.shape)
       
        torch.save(self.label_mat, self.dataset_config.save_label_local_path)
        if self.local is False:
            try:
                save_data(self.dataset_config.save_label_oss_path, self.dataset_config.save_label_local_path)
            except Exception as e:
                print(e)

    def load_label(self):
        '''
        load label data from oss
        '''
        if self.local is False:
            download_file(self.dataset_config.save_label_oss_path, self.dataset_config.save_label_local_path)
        self.logger.info('load label')
        self.label_mat = torch.load(self.dataset_config.save_label_local_path)

    def get_adj_matrix(self):
        label_num = len(self.label_csv)
        print('label num length is', label_num)
        parent_adj = torch.zeros((label_num, label_num))
        child_adj = torch.zeros((label_num, label_num))

        for index, row in self.label_csv.iterrows():
            label_index = int(row['id'])
            parent_adj[label_index, label_index] = 1  # parent matrix 自己和自己设为1
            child_adj[label_index, label_index] = 1  # child matrix 自己和自己设为1
            if pd.isna(row['parent']):
                continue
            else:
                parent_index = int(row['parent'])
                parent_adj[label_index, parent_index] = 1  # parent matrix中自己的行的parent列设为1
                child_adj[parent_index, label_index] = 1  # child matrix中自己设为parent的child

        # result_dict = {'parent_adj': parent_adj, 'child_adj': child_adj}
        self.parent_adj = parent_adj
        print('parent adj', parent_adj)
        self.child_adj = child_adj
        print('child adj', child_adj)

    def deal_label(self, df):
        '''
        df: dataset with label and text
        '''
        # deal with label data
        y_data = list(df['label'])
        if self.label:
            # 获取邻接矩阵
            self.get_adj_matrix()
            # 处理label matrix
            self.init_label()
        # 获取label 独热向量
        self.get_y_data(y_data)
        
        return y_data

    def data_augmentation(self, text):
        if self.da_type == 'eda':
            da_result = eda(text, num_aug=1)[0]
        else:
            raise KeyError
        return da_result

    def deal_x_data(self, df, y_data):
        # deal with x data
        if self.init:
            self.x_text = list(df['data'])
            self.x_data = []
            self.x_da = []
            valid_y = []
            for idx in range(len(self.x_text)):
                # 去掉没有label的数据
                if idx % 1000 == 0:
                    print(idx)
                text = self.x_text[idx]
                # abletion study: da and contrastive loss
                try:
                    da_text = self.data_augmentation(text)
                except:
                    continue
                self.x_data.append(text)
                self.x_da.append(da_text)
                valid_y.append(y_data[idx])
            new_df = pd.DataFrame()
            new_df['data'] = self.x_data
            new_df['da_data'] = self.x_da
            new_df['label'] = valid_y
            new_df.to_csv(self.dataset_config.local_path[self.data_type])
            try:
                save_data(self.dataset_config.oss_path[self.data_type], self.dataset_config.local_path[self.data_type])
            except Exception as e:
                print(e)
        # read data
        df = pd.read_csv(self.dataset_config.local_path[self.data_type])
        # debug: only use 2 batch size data to fast debug
        self.x_data = list(df['data'])
        self.x_da = list(df['da_data'])

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index], self.x_da[index]

    def __len__(self):
        return len(self.x_data)


class KESU_Idx_Dataset(KESU_Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, data_type, emb_layer, logger, label=False, init=False, debug=False, local=True):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        super().__init__(dataset_config, data_type, emb_layer, logger, label=label, init=init, debug=debug, local=local)

    def tokenize(self, text):
        idx_list, padding_mask = self.emb_layer.emb_model.tokenize(text, unk=False)
        return idx_list, padding_mask

    def deal_x_data(self, df, y_data):
        # deal with x data
        if self.init:
            self.x_text = list(df['data'])
            self.x_data = []
            self.x_da = []
            valid_y = []
            for idx in range(len(self.x_text)):
                # 去掉没有label的数据
                if idx % 1000 == 0:
                    print(idx)
                text = self.x_text[idx]

                # abletion study: da and contrastive loss
                try:
                    da_text = self.data_augmentation(text)
                except:
                    continue
                self.x_data.append(text)
                self.x_da.append(da_text)
                valid_y.append(y_data[idx])
            assert len(self.x_data) == len(self.x_da), 'length not equal'
            assert len(self.x_data) == len(valid_y), 'length not equal'
            new_df = pd.DataFrame()
            new_df['data'] = self.x_data
            new_df['da_data'] = self.x_da
            new_df['label'] = valid_y
            new_df.to_csv(self.dataset_config.local_path[self.data_type])
            try:
                save_data(self.dataset_config.oss_path[self.data_type], self.dataset_config.local_path[self.data_type])
            except Exception as e:
                print(e)
        # read data
        df = pd.read_csv(self.dataset_config.local_path[self.data_type])
        
        # debug: only use 2 batch size data to fast debug
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        self.x_data = []
        self.x_da = []
        max_len = len(list(df['data']))
        df_data = list(df['data'])
        df_da_data = list(df['da_data'])
        for i in range(max_len):
            # 去掉没有label的数据
            if i % 1000 == 0:
                print(i)
            x = df_data[i]
            x_da = df_da_data[i]
            try:
                idx_list, padding_mask = self.tokenize(x)
                idx_list_da, padding_mask_da = self.tokenize(x_da)
            except:
                continue
            self.x_idx.append(idx_list)
            self.x_pad.append(padding_mask)
            self.x_data.append(x)
            self.x_da_idx.append(idx_list_da)
            self.x_da_pad.append(padding_mask_da)
            self.x_da.append(x_da)

    def __getitem__(self, index: int):
        result_dict = {
            'origin': {
                'text': self.x_data[index],
                'idx': self.x_idx[index],
                'pad': self.x_pad[index],
                },
            'da': {
                'text': self.x_da[index],
                'idx': self.x_da_idx[index],
                'pad': self.x_da_pad[index],
            },
            'y_data': self.y_data[index]
        }
        return result_dict


class KESU_Idx_Entailment_Dataset(KESU_Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, data_type, emb_layer, logger, label=False, init=False, debug=False, local=True):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        super().__init__(dataset_config, data_type, emb_layer, logger, label=label, init=init, debug=debug, local=local)
        # [tmp_sent_list, y_data, input_ids, attention_mask]
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        self.x_data = []
        self.x_da = []
        self.y_data = []

        if data_type == 'train':
            for i in range(0, dataset_config.train_npy_length):
                number = i * 50000
                print('read file %s' % dataset_config.local_path[data_type + '_' + str(number)])
                result = np.load(dataset_config.local_path[data_type + '_' + str(number)], allow_pickle=True)
                result_da = np.load(dataset_config.local_path[data_type + '_da_' + str(number)], allow_pickle=True)
                for i in result[2]:
                    self.x_idx.append(torch.tensor(i))
                for i in result[3]:
                    self.x_pad.append(torch.tensor(i))
                for i in result_da[2]:
                    self.x_da_idx.append(torch.tensor(i))
                for i in result_da[3]:
                    self.x_da_pad.append(torch.tensor(i))
                for i in result[0]:
                    self.x_data.append(i)
                for i in result_da[0]:
                    self.x_da.append(i)
                for i in result[1]:
                    self.y_data.append(i)

        elif data_type == 'test':
            for i in range(0, dataset_config.test_npy_length):
                number = i * 50000
                result = np.load(dataset_config.local_path[data_type + '_' + str(number)], allow_pickle=True)
                result_da = np.load(dataset_config.local_path[data_type + '_da_' + str(number)], allow_pickle=True)
                for i in result[2]:
                    self.x_idx.append(torch.tensor(i))
                for i in result[3]:
                    self.x_pad.append(torch.tensor(i))
                for i in result_da[2]:
                    self.x_da_idx.append(torch.tensor(i))
                for i in result_da[3]:
                    self.x_da_pad.append(torch.tensor(i))
                for i in result[0]:
                    self.x_data.append(i)
                for i in result_da[0]:
                    self.x_da.append(i)
                for i in result[1]:
                    self.y_data.append(i)

        elif data_type == 'eval':
            for i in range(0, dataset_config.eval_npy_length):
                number = i * 50000
                result = np.load(dataset_config.local_path[data_type + '_' + str(number)], allow_pickle=True)
                result_da = np.load(dataset_config.local_path[data_type + '_da_' + str(number)], allow_pickle=True)
                for i in result[2]:
                    self.x_idx.append(torch.tensor(i))
                for i in result[3]:
                    self.x_pad.append(torch.tensor(i))
                for i in result_da[2]:
                    self.x_da_idx.append(torch.tensor(i))
                for i in result_da[3]:
                    self.x_da_pad.append(torch.tensor(i))
                for i in result[0]:
                    self.x_data.append(i)
                for i in result_da[0]:
                    self.x_da.append(i)
                for i in result[1]:
                    self.y_data.append(i)
        self.fake_y_data = [-1 for _ in self.y_data]
        self.index_list = [i for i in range(len(self.y_data))]
        
        self.backup_x_data = self.x_data
        self.backup_x_idx = self.x_idx
        self.backup_x_pad = self.x_pad
        self.backup_x_da = self.x_da
        self.backup_x_da_idx = self.x_da_idx
        self.backup_x_da_pad = self.x_da_pad
        self.backup_y_data = self.y_data
        self.backup_fake_y_data = self.fake_y_data
        self.backup_index_list = self.index_list

        print('read end', len(self.x_idx), len(self.x_da_idx), len(self.x_data), len(self.y_data))

    def tokenize(self, texts):
        idx_list, padding_mask = self.emb_layer.emb_model.tokenizer_sentences(texts, unk=False)
        # idx_list = [torch.tensor(i) for i in idx_list]
        # padding_mask = [torch.tensor(i) for i in padding_mask]
        return idx_list, padding_mask

    def deal_x_data(self, df, y_data):
        # deal with x data
        if self.init:
            self.x_text = list(df['data'])
            self.x_data = []
            self.x_da = []
            valid_y = []
            for idx in range(len(self.x_text)):
                # 去掉没有label的数据
                if idx % 1000 == 0:
                    print(idx)
                text = self.x_text[idx]

                # abletion study: da and contrastive loss
                try:
                    da_text = self.data_augmentation(text)
                except:
                    continue
                self.x_data.append(text)
                self.x_da.append(da_text)
                valid_y.append(y_data[idx])
            assert len(self.x_data) == len(self.x_da), 'length not equal'
            assert len(self.x_data) == len(valid_y), 'length not equal'
            new_df = pd.DataFrame()
            new_df['data'] = self.x_data
            new_df['da_data'] = self.x_da
            new_df['label'] = valid_y
            new_df.to_csv(self.dataset_config.local_path[self.data_type])
            try:
                save_data(self.dataset_config.oss_path[self.data_type], self.dataset_config.local_path[self.data_type])
            except Exception as e:
                print(e)
            
    def __getitem__(self, index: int):
        result_dict = {
            'origin': {
                'text': self.x_data[index],
                'idx': self.x_idx[index],
                'pad': self.x_pad[index],
                },
            'da': {
                'text': self.x_da[index],
                'idx': self.x_da_idx[index],
                'pad': self.x_da_pad[index],
            },
            'y_data': self.y_data[index],
            'fake_y_data': self.fake_y_data[index],
            'index': self.index_list[index]
        }
        return result_dict

    def rebuild_fake_label_dataset(self, labeled_index_list:list):
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        self.x_data = []
        self.x_da = []
        self.y_data = []
        self.index_list = []
        self.fake_y_data = []

        for i in range(len(self.backup_index_list)):
            index = self.backup_index_list[i]
            if labeled_index_list[index] != -1:
                self.x_idx.append(self.backup_x_idx[i])
                self.x_pad.append(self.backup_x_pad[i])
                self.x_da_idx.append(self.backup_x_da_idx[i])
                self.x_da_pad.append(self.backup_x_da_pad[i])
                self.x_data.append(self.backup_x_data[i])
                self.x_da.append(self.backup_x_da[i])
                self.y_data.append(self.backup_y_data[i])
                self.index_list.append(self.backup_index_list[i])
                self.fake_y_data.append(labeled_index_list[index])

        print('Successfully rebuild the dataset and remove labeled data, length is', len(self.x_data), len(self.backup_x_data))

    def rebuild_no_fake_label_dataset(self, labeled_index_list: list):
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        self.x_data = []
        self.x_da = []
        self.y_data = []
        self.index_list = []
        self.fake_y_data = []

        for i in range(len(self.backup_index_list)):
            index = self.backup_index_list[i]
            if labeled_index_list[i] == -1:
                self.x_idx.append(self.backup_x_idx[i])
                self.x_pad.append(self.backup_x_pad[i])
                self.x_da_idx.append(self.backup_x_da_idx[i])
                self.x_da_pad.append(self.backup_x_da_pad[i])
                self.x_data.append(self.backup_x_data[i])
                self.x_da.append(self.backup_x_da[i])
                self.y_data.append(self.backup_y_data[i])
                self.index_list.append(self.backup_index_list[i])
                self.fake_y_data.append(self.backup_fake_y_data[i])

        print('Successfully rebuild the dataset and remove labeled data, length is', len(self.x_data), len(self.backup_x_data))

    def rebuild_all_dataset(self):
        self.x_idx = self.backup_x_idx
        self.x_pad = self.backup_x_pad
        self.x_da_idx = self.backup_x_da_idx
        self.x_da_pad = self.backup_x_da_pad
        self.x_data = self.backup_x_data
        self.x_da = self.backup_x_da
        self.y_data = self.backup_y_data
        self.index_list = self.backup_index_list
        self.fake_y_data = self.backup_fake_y_data
        print('Successfully recover the dataset, length is', len(self.x_data))


class KESU_Idx_Label_dataset(KESU_Dataset):
    '''
    generated class and result
    '''
    def __init__(self, dataset_config, data_type, emb_layer, logger, label=False, init=True, debug=False, local=True):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        super().__init__(dataset_config, data_type, emb_layer, logger, label=label, init=init, debug=debug, local=local)
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        label_csv = pd.read_csv(dataset_config.label_local_path)
        self.y_text = list(label_csv['description'])
        self.y_label = list(label_csv['id'])
        if self.init:
            self.y_da_text = []
            for idx in range(len(self.y_text)):
                # 去掉没有label的数据
                text = self.y_text[idx]
                # abletion study: da and contrastive loss
                da_text = self.data_augmentation(text)
                self.y_da_text.append(da_text)
            
            label_csv['da_description'] = self.y_da_text
            label_csv.to_csv(dataset_config.label_local_path)
            try:
                save_data(self.dataset_config.label_oss_path, dataset_config.label_local_path)
            except Exception as e:
                print(e)

        print('y_label length', len(self.y_text))
        label_csv = pd.read_csv(dataset_config.label_local_path)
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        self.x_data = list(label_csv['description'])
        self.x_da = list(label_csv['description'])
        for x in self.x_data:
            idx_list, padding_mask = self.tokenize(x)
            self.x_idx.append(idx_list)
            self.x_pad.append(padding_mask)
        for x in self.x_da:
            idx_list, padding_mask = self.tokenize(x)
            self.x_da_idx.append(idx_list)
            self.x_da_pad.append(padding_mask)

    def tokenize(self, text):
        idx_list, padding_mask = self.emb_layer.emb_model.tokenize(text, unk=False)
        return idx_list, padding_mask

    def __getitem__(self, index: int):
        result_dict = {
            'origin': {
                'text': self.x_data[index],
                'idx': self.x_idx[index],
                'pad': self.x_pad[index],
                },
            'da': {
                'text': self.x_da[index],
                'idx': self.x_da_idx[index],
                'pad': self.x_da_pad[index],
            },
            'y_data': self.y_label[index]
        }
        return result_dict

    def __len__(self):
        return len(self.y_label)


class KESU_Generated_Dataset(Dataset):
    '''
    generated class and result
    '''
    def __init__(self, class_num, dataset_config):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.class_num = class_num
        label_csv = pd.read_csv(dataset_config.label_local_path)
        self.y_text = list(label_csv['label'])
        print('y_label length', len(self.y_text), class_num)
        self.x = []
        self.y = []
        for i in range(class_num):
            self.x.append(torch.tensor(i))
            self.y.append(torch.tensor(i))

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.y_text[index]

    def __len__(self):
        return len(self.x)


class KESU_Generated_Enailment_Dataset(Dataset):
    '''
    generated class and result
    '''
    def __init__(self, class_num, dataset_config):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.class_num = class_num
        label_csv = pd.read_csv(dataset_config.label_local_path)
        tmp_label_list = [str(label_name) for label_name in list(label_csv['label'])]
        label_list = []
        maintain_length = 2
        for label in tmp_label_list:
            label = label.split('/')
            if len(label) < maintain_length:
                label_list.append('/'.join(label))
            else:
                label = label[-maintain_length:]
                label_list.append('/'.join(label))
        self.y_text = ['关于%s的客户投诉，客户说' % str(label_name) for label_name in list(label_csv['label'])]
        self.y_true_text = [label_name for label_name in list(label_csv['label'])]
        print('y_label length', len(self.y_text), class_num)
        self.x = []
        self.y = []
        for i in range(class_num):
            self.x.append(torch.tensor(i))
            self.y.append(torch.tensor(i))


    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.y_text[index], self.y_true_text[index]

    def __len__(self):
        return len(self.x)


class KESU_Unlabeled_Dataset(Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, emb_layer, debug=False, local=False, init=False):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        self.local = local  # local or mayi server
        self.dataset_config = dataset_config  # config
        self.da_type = dataset_config.da_type  # data augmentation type
        self.emb_layer = emb_layer  # embedding layer
        # download unlabeled data
        df = pd.read_csv(dataset_config.unlabel_local_path, sep=',')
        self.text = list(df['data'])

        # start data augmentation
        if init:
            self.text_da = []
            self.text_choosed = []
            for idx in tqdm(range(len(self.text))):
                text = self.text[idx]
                # abletion study: da and contrastive loss
                try:
                    da_text = self.data_augmentation(text)
                except:
                    continue
                self.text_choosed.append(text)
                self.text_da.append(da_text)
            new_df = pd.DataFrame()
            new_df['data'] = self.text_choosed
            new_df['da_data'] = self.text_da
            new_df.to_csv(dataset_config.unlabel_local_path)
            try:
                save_data(dataset_config.unlabel_oss_path, dataset_config.unlabel_local_path)
            except Exception as e:
                print(e)

        df = pd.read_csv(dataset_config.unlabel_local_path)
        # debug: only use 2 batch size data to fast debug
        self.text = list(df['data'])
        self.text_da = list(df['da_data'])

    def data_augmentation(self, text):
        if self.da_type == 'eda':
            da_result = eda(text, num_aug=1)[0]
        else:
            raise KeyError
        return da_result

    def __getitem__(self, index: int):
        return self.text[index], self.text_da[index]

    def __len__(self):
        return len(self.text)


class KESU_Idx_Unlabeled_Dataset(KESU_Unlabeled_Dataset):
    """
    dataset类
    """
    def __init__(self, dataset_config, emb_layer, debug=False, local=False, init=False):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        super().__init__(dataset_config, emb_layer, debug=debug, local=local, init=init)
        self.clean_text = []
        self.clean_text_da = []
        self.x_idx = []
        self.x_pad = []
        self.x_da_idx = []
        self.x_da_pad = []
        for i in range(len(self.text)):
            try:
                x = self.text[i]
                x_da = self.text_da[i]
                # print(x_da)
                try:
                    idx_list, padding_mask = self.tokenize(x)
                    da_idx_list, da_padding_mask = self.tokenize(x_da)
                except:
                    continue
                
                self.clean_text.append(x)
                self.clean_text_da.append(x_da)
                self.x_idx.append(idx_list)
                self.x_pad.append(padding_mask)
                self.x_da_idx.append(da_idx_list)
                self.x_da_pad.append(da_padding_mask)
            except AssertionError as e:
                print(e, i, self.text[i], self.text_da[i])


    def tokenize(self, text):
        idx_list, padding_mask = self.emb_layer.emb_model.tokenize(text, unk=False)
        return idx_list, padding_mask

    def __getitem__(self, index: int):
        result_dict = {
            'origin': {
                'text': self.clean_text[index],
                'idx': self.x_idx[index],
                'pad': self.x_pad[index],
                },
            'da': {
                'text': self.clean_text_da[index],
                'idx': self.x_da_idx[index],
                'pad': self.x_da_pad[index],
            },
        }
        return result_dict
    
    def __len__(self):
        return len(self.clean_text)
