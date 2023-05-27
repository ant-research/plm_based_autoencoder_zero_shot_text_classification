from model.new_model.modules.encoder.bert_encoder import Bert_CLS, Bert_Pooler, Bert_Entail_CLS
from model.new_model.modules.encoder.encoder import TransformerQuantizerEncoder, Transformer_Disperse


Encoder = {
    # 'Transformer': Transformer,
    'Transformer': TransformerQuantizerEncoder,
    'Transformer_P': Transformer_Disperse,
    'Bert_CLS': Bert_CLS,
    'Bert_Pooler': Bert_Pooler,
    'Bert_Entail_CLS': Bert_Entail_CLS,
}
