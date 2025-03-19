import torch.nn as nn
from models.attention import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        # Self-Attention + Residual + Norm
        attn_out = self.self_attn(src, src, src, mask)
        src = self.norm1(src + attn_out)
        # FFN + Residual + Norm
        ff_out = self.ffn(src)
        src = self.norm2(src + ff_out)
        return src
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, mask=None):
        out = self.embedding(src)
        out = self.pos_encoding(out)

        for layer in self.layers:
            out = layer(out, mask)

        logits = self.fc_out(out)
        return logits