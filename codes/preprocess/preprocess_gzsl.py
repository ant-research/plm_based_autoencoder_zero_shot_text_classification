# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
import sys
import pickle
import shutil

import numpy as np
import pandas
import tqdm
import re
from codes.config import GZSLConfig
from codes.utils.download import download_mimic3


def log(s, f=sys.stderr):
    if s[-1] != '\n':
        s += '\n'
    f.write(s)
    f.flush()


class Tokenizer(object):
    def __init__(self, vocab: dict):
        super(Tokenizer, self).__init__()
        self.vocab = vocab
        self.special_rep = 'NOUN'
        self.unk = "<UNK>"
        self.num = "<NUM>"
        self.pat = re.compile(r'(\[\*\*[^\[]*\*\*\])')

    def remove_special_token(self, sent):
        return self.pat.sub(self.special_rep, sent)

    @staticmethod
    def tokenize(sent):
        words = [s for s in re.split(r"\W+", sent) if s and not s.isspace()]
        return words

    def replace_unknowns_nums(self, words):
        tokens = []
        for word in words:
            if self.special_rep.lower() == word:
                continue

            if word in self.vocab:
                tokens.append(word)
            else:
                token = self.distinguish_unk_num(word)
                if len(tokens) == 0 or tokens[-1] != token:
                    tokens.append(token)
        return tokens

    def distinguish_unk_num(self, word):
        if self.only_numerals(word):
            return self.num
        else:
            return self.unk

    @staticmethod
    def only_numerals(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def process(self, input_text: str):
        input_text = self.remove_special_token(input_text)
        words_tokenized = self.tokenize(input_text)
        words = [word.lower().strip() for word in words_tokenized]
        words = self.replace_unknowns_nums(words)
        return words

    



def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def remove_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def is_discharge_summary(note_category):
    return 'discharge summary' in note_category.lower().strip()


def get_patient_data(mimic_dir):
    read_file = f'{mimic_dir}/NOTEEVENTS.csv'
    log(f'Reading {read_file} ...')
    df_notes = pandas.read_csv(read_file, low_memory=False, dtype=str)

    read_file = f'{mimic_dir}/DIAGNOSES_ICD.csv'
    log(f'Reading {read_file} ...')
    df_icds = pandas.read_csv(read_file, low_memory=False, dtype=str)

    all_notes = df_notes['TEXT']
    all_note_types = df_notes['CATEGORY']
    all_note_descriptions = df_notes['DESCRIPTION']

    subject_ids_notes = df_notes['SUBJECT_ID']
    hadm_ids_notes = df_notes['HADM_ID']

    subject_ids_icd = df_icds['SUBJECT_ID']
    hadm_ids_icd = df_icds['HADM_ID']
    seq_nums_icd = df_icds['SEQ_NUM']
    icd9_codes = df_icds['ICD9_CODE']
    patient_dict = {(subject_id, hadm_id): [{}, {}] for subject_id, hadm_id in zip(subject_ids_notes, hadm_ids_notes)}

    # staring with icd code labels and collecting only those subject_id,
    # hadm_id pairs with at least one non-nan icd label
    for (subject_id, hadm_id, seq_num, icd9_code) in zip(subject_ids_icd, hadm_ids_icd, seq_nums_icd, icd9_codes):
        try:  # there are cases where subject id, hadm id pairs are present in icd code data but not in noteevents data.
            # checking for nan, will fail for string then go to except and put in patient dict
            if not math.isnan(seq_num):
                patient_dict[(subject_id, hadm_id)][1][seq_num] = icd9_code
        except TypeError:
            try:
                patient_dict[(subject_id, hadm_id)][1][seq_num] = icd9_code
            except KeyError:  # if not in admissions data
                pass

    for (subject_id, hadm_id, note, note_type, note_description) in zip(subject_ids_notes, hadm_ids_notes, all_notes,
                                                                        all_note_types, all_note_descriptions):
        if is_discharge_summary(note_type):
            if (note_type, note_description) in patient_dict[(subject_id, hadm_id)][0]:
                patient_dict[(subject_id, hadm_id)][0][(note_type, note_description)].append(note)
            else:
                patient_dict[(subject_id, hadm_id)][0][(note_type, note_description)] = [note]

    to_remove = []
    for (subject_id, hadm_id) in patient_dict:
        if len(patient_dict[(subject_id, hadm_id)][0]) == 0 or len(patient_dict[(subject_id, hadm_id)][1]) == 0:
            to_remove.append((subject_id, hadm_id))
    for key in to_remove:
        patient_dict.pop(key)

    log(f'Total number of (subject_id, hadm_id) with discharge summary, with at least 1 code: {len(patient_dict)}')
    return patient_dict


def concat_and_write(list_of_notes, concatenated_file, tokenizer):
    concatenated_text = ''.join(list_of_notes)
    words = tokenizer.process(concatenated_text)
    print(len(words))
    f = open(concatenated_file, 'w')
    f.write(concatenated_text)
    f.close()


def extract_text_files(mimic_dir, save_dir):
    patient_dict = get_patient_data(mimic_dir)

    text_save_dir = f'{save_dir}/text_files/'
    make_folder(text_save_dir)
    label_save_dir = f'{save_dir}/label_files/'
    make_folder(label_save_dir)

    log(f'Loading vocab dict saved during from {VOCAB_DICT_PATH}')
    with open(VOCAB_DICT_PATH, 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab)

    total_txt_count = 0
    for (subject_id, hadm_id) in tqdm.tqdm(patient_dict, desc='Extracting text files'):
        icd9_dict = patient_dict[(subject_id, hadm_id)][1]

        all_descriptions = []
        for category, description in patient_dict[(subject_id, hadm_id)][0].keys():
            notes = patient_dict[(subject_id, hadm_id)][0][(category, description)]
            all_descriptions.extend(notes)

        # writing description notes
        text_save_path = f'{text_save_dir}/{subject_id}_{hadm_id}_notes.txt'
        print(all_descriptions)
        concat_and_write(all_descriptions, text_save_path, tokenizer)
        # writing icd labels
        label_save_path = f'{label_save_dir}/{subject_id}_{hadm_id}_labels.txt'
        f = open(label_save_path, 'w')
        for key in icd9_dict:
            f.write('{}, {}\n'.format(key, icd9_dict[key]))
        f.close()
        total_txt_count += 1
        break

    log(f'Written {total_txt_count} text files to {save_dir}')


def tokenize_raw_text(save_dir):
    text_save_dir = os.path.join(save_dir, 'text_files')
    numpy_vectors_save_dir = os.path.join(save_dir, 'numpy_vectors')
    remove_folder(numpy_vectors_save_dir)
    make_folder(numpy_vectors_save_dir)
    hadms = []
    for filename in os.listdir(text_save_dir):
        if ".txt" in filename:
            hadm = filename.replace(".txt", "")
            hadms.append(hadm)
    log(f"Total number of text files in set: {len(hadms)}")

    log(f'Loading vocab dict saved during from {VOCAB_DICT_PATH}')
    with open(VOCAB_DICT_PATH, 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab)

    for hadm in tqdm.tqdm(hadms, desc='Tokenizing raw patient notes'):
        text = open(os.path.join(text_save_dir, str(hadm) + ".txt"), "r").read()
        words = tokenizer.process(text)
        vector = []
        for word in words:
            if word in vocab:
                vector.append(vocab[word])
            elif tokenizer.only_numerals(word) and (len(vector) == 0 or vector[-1] != vocab["<NUM>"]):
                vector.append(vocab["<NUM>"])

        mat = np.array(vector)
        # saving word indices to file
        write_file = os.path.join(numpy_vectors_save_dir, f"{hadm}.npy")
        np.save(write_file, mat)


def build_dataset(PROCESSED_DIR):
    modes = ['train', 'test', 'dev']
    thresh = 5
    for mode in modes:
        names = []
        with open(f'./data/resources/splits/{mode}_thresh{thresh}_hdam_ids.txt') as f:
            for line in f:
                names.append(line[:-1])
        names = sorted(names)
        label_paths, note_paths = [], []
        names = names

        for name in names:
            label_paths.append(os.path.join(os.path.join(PROCESSED_DIR, 'label_files/'), f'{name}_labels.txt'))
            note_paths.append(os.path.join(os.path.join(PROCESSED_DIR, 'text_files/'), f'{name}_notes.txt'))

def main():
    global VOCAB_DICT_PATH
    config = GZSLConfig()
    #download_mimic3(config)
    PROCESSED_DIR = config.processed_dir
    VOCAB_DICT_PATH = config.vocab_path
    mimic_dir = config.local_path
    extract_text_files(mimic_dir, PROCESSED_DIR)
    #tokenize_raw_text(PROCESSED_DIR)
    #build_dataset(PROCESSED_DIR)
