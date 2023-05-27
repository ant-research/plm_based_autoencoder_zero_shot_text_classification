class DataArgs:
    '''
    a Args class that maintain all arguments for data preparation
    '''
    data_path = ""
    model_for_labelvocab = "deepset/sentence_bert"
    max_seq_length = 128
    label_field_name = "label"
    bs = 16
    entail_model = "facebook/bart-large-mnli"
    model_for_preselftrain = "bert-base-uncased"
    model_for_fulltrain = "bert-base-uncased"
    multi_label = False
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)