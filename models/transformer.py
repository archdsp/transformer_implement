import torch.nn as nn


from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        pass

