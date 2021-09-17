import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.v = nn.Parameter(torch.rand(self.decoder_dim))
        self.W_1 = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = nn.Linear(self.encoder_dim, self.decoder_dim)

    def forward(self, query, values, mask=None):
        # [seq_length, decoder_dim]
        query = query.repeat(1, values.size(1), 1)

        weights = self.W_1(query) + self.W_2(values)
        weights = torch.tanh(weights) @ self.v

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        return weights
