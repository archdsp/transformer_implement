import torch.nn as nn
from models.attention import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerDecoderLayer, self).__init__()
        pass
        
    def forward(self, tgt, src, src_mask=None, tgt_mask=None):
        pass


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len):
        super().__init__()
        pass

    def forward(self, tgt, src, src_mask=None, tgt_mask=None):
        pass