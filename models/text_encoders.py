from torch import nn


class BiLSTMTextEncoder(nn.Module):
    def __init__(self, embeddings, hdim: int = 128, n_layers: int = 2, freeze_embeddings: bool = False, dropout_p: float = 0.5):
        super().__init__()

        self.edim = embeddings.size(1)
        self.hdim = hdim
        self.n_layers = n_layers
        self.output_size = hdim * 2

        self.embed = nn.Embedding.from_pretrained(
            embeddings, freeze=freeze_embeddings)
        self.lstm = nn.LSTM(
            self.edim,
            self.hdim,
            num_layers=self.n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, inputs):
        X = self.embed(inputs)
        X, _ = self.lstm(X)

        return X
