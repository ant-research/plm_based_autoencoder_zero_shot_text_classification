from cgi import test
import pandas as pd
import numpy as np
import re
from codes.wos.dataset_config import WOSDataConfig, WOSDataConfig_W2V
from codes.utils import download_file, save_data
from keras.preprocessing.text import Tokenizer
import torch
#from torchtext.data import get_tokenizer
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os
import json
import numpy as np
import re

"""
WoS Reference: https://github.com/kk7nc/HDLTex
"""

train_v0_file = '../data/situation/train_pu_half_v0.txt'
train_v1_file = '../data/situation/train_pu_half_v1.txt'
dev_file = '../data/situation/dev.txt'
test_file = '../data/situation/test.txt'
classes_file = '../data/situation/classes.txt'


np.random.seed(7)


def build_label_matrix():
    with open(classes_file, 'r') as f:
        data_list = f.readlines()
    label_list = [label[:-1] for label in data_list]
    
    label_df = pd.DataFrame()
    label_id_list = []
    label_parent_list = []
    label_name_list = []
    # deal with label parent
    for i in range(len(label_list)):
        label_id_list.append(i)
        label_name_list.append(label_list[i])
        label_parent_list.append(np.nan)
    label_df['id'] = label_id_list
    label_df['label'] = label_name_list
    label_df['parent'] = label_parent_list

    label_df.to_csv('../data/situation/label.csv', index=False)
    return label_df
        

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    # string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # words = string.split(' ')
    # final_words = []
    # for word in words:
    #     if word.lower() in english_stopwords:
    #         pass
    #     else:
    #         final_words.append(word)
    # string = ' '.join(final_words)
    return string.strip().lower()


def convert_file_to_csv(data_list):
    label_list = []
    text_list = []
    for data in data_list:
        text = data.split('\t')[-1][:-1]
        clear_text = clean_str(text)
        text_list.append(clear_text)
        label_list.append(data.split('\t')[0])
    df = pd.DataFrame()
    df['label'] = label_list
    df['data'] = text_list
    return df


def get_all_classes():
    with open(test_file, 'r') as f:
        data_list = f.readlines()
    df = convert_file_to_csv(data_list)
    labels_list = df['label'].unique()
    with open(classes_file, 'w') as f:
        for label in labels_list:
            f.write(label + '\n')


def sample_train_data_from_meta(sample=False):
    label_csv = pd.read_csv('../data/situation/label.csv')
    with open(train_v0_file, 'r') as f:
        data_list = f.readlines()
    df = convert_file_to_csv(data_list)
    new_label_list = []
    new_text_list = []
    total_label_list = list(label_csv['label'])
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        label = row['label']
        if label in total_label_list:
            new_text_list.append(row['data'])
            new_label_list.append(total_label_list.index(label))
    new_df['data'] = new_text_list
    new_df['label'] = new_label_list
    df = new_df
        
    new_df = pd.DataFrame()
    for i in df['label'].unique():
        if sample:
            tmp = df[df['label'] == i].sample(frac=0.1)
            new_df = new_df.append(tmp)
        else:
            new_df = new_df.append(df[df['label'] == i])
    new_df.to_csv('../data/situation/trainv0.csv')
    
    
    with open(train_v1_file, 'r') as f:
        data_list = f.readlines()
    df = convert_file_to_csv(data_list)
    new_label_list = []
    new_text_list = []
    total_label_list = list(label_csv['label'])
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        label = row['label']
        if label in total_label_list:
            new_text_list.append(row['data'])
            new_label_list.append(total_label_list.index(label))
    new_df['data'] = new_text_list
    new_df['label'] = new_label_list
    df = new_df
    new_df = pd.DataFrame()
    for i in df['label'].unique():
        if sample:
            tmp = df[df['label'] == i].sample(frac=0.1)
            new_df = new_df.append(tmp)
        else:
            new_df = new_df.append(df[df['label'] == i])
    new_df.to_csv('../data/situation/trainv1.csv')


def get_data_from_meta():
    label_csv = pd.read_csv('../data/situation/label.csv')
    sample_train_data_from_meta()
    with open(dev_file, 'r') as f:
        data_list = f.readlines()
    df = convert_file_to_csv(data_list)
    new_label_list = []
    new_text_list = []
    total_label_list = list(label_csv['label'])
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        label = row['label']
        if label in total_label_list:
            new_text_list.append(row['data'])
            new_label_list.append(total_label_list.index(label))
    new_df['data'] = new_text_list
    new_df['label'] = new_label_list
    df = new_df
    df.to_csv('../data/situation/eval.csv')
    
    with open(test_file, 'r') as f:
        data_list = f.readlines()
    df = convert_file_to_csv(data_list)
    new_label_list = []
    new_text_list = []
    total_label_list = list(label_csv['label'])
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        label = row['label']
        if label in total_label_list:
            new_text_list.append(row['data'])
            new_label_list.append(total_label_list.index(label))
    new_df['data'] = new_text_list
    new_df['label'] = new_label_list
    df = new_df
    df.to_csv('../data/situation/test.csv')
    print(df['label'].unique())
    df.to_csv('../data/situation/unlabeled.csv')
    build_label_matrix()


def split_train_dev_test():
    f = open('wos_total.json', 'r')
    data = f.readlines()
    f.close()
    id = [i for i in range(46985)]
    np_data = np.array(data)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    f = open('wos_train.json', 'w')
    f.writelines(train)
    f.close()
    f = open('wos_test.json', 'w')
    f.writelines(test)
    f.close()
    f = open('wos_val.json', 'w')
    f.writelines(val)
    f.close()

    print(len(train), len(val), len(test))
    return


if __name__ == '__main__':
    get_data_from_meta()
    split_train_dev_test()