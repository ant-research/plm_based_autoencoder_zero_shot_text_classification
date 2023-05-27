from turtle import forward
import torch
import torch.nn as nn
from model.discrete_vae.models.transformer import TransformerEncoderDecoder, TransformerQuantizerEncoder


class Model(nn.Module):
    def __init__(self, config, emb_layer) -> None:
        super().__init__()
        encoder = TransformerQuantizerEncoder(config=config, emb_layer=emb_layer)
        self.model = TransformerEncoderDecoder(config=config, encoder=encoder, emb_layer=emb_layer)
        
    def forward(self, inputs):
        result = self.model(inputs)
        return result
