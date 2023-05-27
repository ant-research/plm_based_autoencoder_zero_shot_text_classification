from model.new_model.modules.decoder.transformer_decoder import TransformerDecoder, TransformerDecoderDistance
from model.new_model.modules.decoder.gpt2 import NewGPT
from model.new_model.modules.decoder.gpt import (
    GPT, GPT_EASY, GPT_Emb_EASY, GPT_Emb_Match_Network_EASY, GPT_Emb_Match_Network_EASY_Concat3,
    GPT_Emb_Match_Network_EASY_Concat2, GPT_EASY_DVQ, GPT_Emb_Match_Network_DVQ
)
from model.new_model.modules.decoder.dilated_cnn import DilatedCNN

Decoder = {
    'DilatedCNN': DilatedCNN,
    'Transformer': TransformerDecoder,
    'TransformerDistance': TransformerDecoderDistance,
    'GPT': GPT,
    'GPT_EASY': GPT_EASY,
    'GPT_EMB_EASY': GPT_Emb_EASY,
    'GPT_Match': GPT_Emb_Match_Network_EASY,
    'GPT_Match3': GPT_Emb_Match_Network_EASY_Concat3,
    'GPT_Match2': GPT_Emb_Match_Network_EASY_Concat2,
    'GPT_EASY_DVQ': GPT_EASY_DVQ,
    'GPT_Match_DVQ': GPT_Emb_Match_Network_DVQ,
    'GPT2': NewGPT
}
