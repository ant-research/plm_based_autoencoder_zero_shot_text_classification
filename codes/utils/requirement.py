# -*- coding: utf-8 -*-
'''
@Author: guokaihao.gkh
@Describe: 安装内网下不到的包
'''

import os
import zipfile
from codes.config import RequirementConfig
from codes.utils.download import download_file


def install_wordnet(local=True):
    
    config = RequirementConfig()
    local_path = config.wordnet_local_path
    if local is False:
        oss_path = config.wordnet_oss_path
        download_file(oss_path, local_path)

    f = zipfile.ZipFile(local_path)
    pkgs_path = './data/pkgs'
    if os.path.exists(pkgs_path):
        pass
    else:
        os.mkdir(pkgs_path)
    corpora_path = './data/pkgs/corpora'
    if os.path.exists(corpora_path):
        pass
    else:
        os.mkdir(corpora_path)
    for file in f.namelist():
        f.extract(file, './data/pkgs/corpora')
    f.close()


def install_stopwords(local=False):
    config = RequirementConfig()
    local_path = config.stopwords_local_path
    oss_path = config.stopwords_oss_path
    download_file(oss_path, local_path)

    f = zipfile.ZipFile(local_path)
    pkgs_path = './data/pkgs'
    if os.path.exists(pkgs_path):
        pass
    else:
        os.mkdir(pkgs_path)
    corpora_path = './data/pkgs/corpora'
    if os.path.exists(corpora_path):
        pass
    else:
        os.mkdir(corpora_path)
    for file in f.namelist():
        f.extract(file, './data/pkgs/corpora')
    f.close()
    

def install_owms(local=False):
    config = RequirementConfig()
    local_path = config.owm_local_path
    oss_path = config.owm_oss_path
    download_file(oss_path, local_path)

    f = zipfile.ZipFile(local_path)
    pkgs_path = './data/pkgs'
    if os.path.exists(pkgs_path):
        pass
    else:
        os.mkdir(pkgs_path)
    corpora_path = './data/pkgs/corpora'
    if os.path.exists(corpora_path):
        pass
    else:
        os.mkdir(corpora_path)
    for file in f.namelist():
        f.extract(file, './data/pkgs/corpora')
    f.close()



def install_synonyms(local=False):
    os.environ['SYNONYMS_WORD2VEC_BIN_MODEL_ZH_CN'] = './data/pkgs/words.vector'
    config = RequirementConfig()
    local_path = config.synonyms_local_path
    oss_path = config.synonyms_oss_path
    download_file(oss_path, local_path)


def main():
    install_wordnet()
    install_stopwords()


if __name__ == '__main__':
    main()