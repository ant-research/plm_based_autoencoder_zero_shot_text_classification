# from model.new_model.train import OneModelTrainer, OneModelTrainerParallel
from model.new_model.train_entailment_no_dis import (
    OneModelTrainerParallel_Entailment,
    OneModelTrainerParallel_EntailmentImbalanceSemi
)

from model.new_model.train_entailmen_dis import (
    OneModelTrainerParallel_Entailment_Dis,
    OneModelTrainerParallel_EntailmentImbalance_Dis,
    OneModelTrainerParallel_EntailmentImbalanceSemi_Dis
)
from model.gpt2model.train import OneModelTrainer_GPT2, OneModelTrainerParallel_GPT2
from model.bertmodel.train import OneModelTrainer_BERT, OneModelTrainerParallel_Bert
from model.transicd.train import OneModelTrainer_TransICD, OneModelTrainerParallel_TransICD
from model.fzml.train import OneModelTrainer_MLZS, OneModelTrainerParallel_MLZS
from model.attentionxml.train import OneModelTrainer_AttnXML, OneModelTrainerParallel_AttnXML
from model.easy_sim.train import OneModelTrainer_SIM, OneModelTrainerParallel_SIM
from model.zerogen.train import OneModelTrainer_ZeroGen, OneModelTrainerParallel_ZeroGen
from model.bert_rl.train import OneModelTrainer_RL


ModelTrainer = {
    'w2v': {
        'TransICD': OneModelTrainer_TransICD,
        'FZML': OneModelTrainer_MLZS,
        'AttnXML': OneModelTrainer_AttnXML,
        'EASYSIM': OneModelTrainer_SIM,
        'GPT2Classifier': OneModelTrainer_GPT2,
        'BERTClassifier': OneModelTrainer_BERT
    },
    'bert': {
        'GPT2Classifier': OneModelTrainer_GPT2,
        'BERTClassifier': OneModelTrainer_BERT,
        'ZeroGen': OneModelTrainer_ZeroGen,
        'Bert_RL': OneModelTrainer_RL
    }
}

ModelTrainerParallel = {
    'w2v': {
        'TransICD': OneModelTrainerParallel_TransICD,
        'FZML': OneModelTrainerParallel_MLZS,
        'AttnXML': OneModelTrainerParallel_AttnXML,
        'EASYSIM': OneModelTrainerParallel_SIM,
        'GPT2Classifier': OneModelTrainerParallel_GPT2,
        'BERTClassifier': OneModelTrainerParallel_Bert,
    },
    'bert': {
        'GPT2Classifier': OneModelTrainerParallel_GPT2,
        'BERTClassifier': OneModelTrainerParallel_Bert,
        'Entailment': OneModelTrainerParallel_EntailmentImbalanceSemi_Dis,
        'ZeroGen': OneModelTrainerParallel_ZeroGen
    }
}