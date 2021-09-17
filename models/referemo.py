import torch
from torch import nn
from utils import ReferEmoModelInput

from models.attention import Attention
from models.reference_encoders import BERTReferenceEncoder
from models.text_encoders import BiLSTMTextEncoder


# Model hyperparameters are
# pretrained_bert = "bert-base-uncased"
# hdim = 128
# n_layers = 2
# freeze_embeddings = False
# encoder_dropout_p = 0.5
# classification_dropout_p = 0.5
class ReferEmo(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        nclasses: int,
        pretrained_bert: str = 'bert-base-uncased',
        hdim: int = 128,
        n_layers: int = 2,
        freeze_embeddings: bool = False,
        encoder_dropout_p: float = 0.5,
        classification_dropout_p: float = 0.5
    ):
        super().__init__()

        self.reference_encoder = BERTReferenceEncoder(pretrained_bert)
        self.text_encoder = BiLSTMTextEncoder(
            embeddings,
            hdim=hdim,
            n_layers=n_layers,
            freeze_embeddings=freeze_embeddings,
            dropout_p=encoder_dropout_p
        )
        self.attention = Attention(
            self.text_encoder.output_size,
            self.reference_encoder.output_size
        )

        hidden_size = self.text_encoder.output_size

        self.linear_two = nn.Linear(hidden_size, hidden_size // 2)
        self.linear_one = nn.Linear(hidden_size // 2, nclasses)

        self.dropout = nn.Dropout(classification_dropout_p)

    def forward(self, model_input: ReferEmoModelInput):
        ref_input = model_input.reference_inputs
        text_input = model_input.text_inputs
        text_masks = model_input.text_masks

        ref_enc, _ = self.reference_encoder(ref_input)
        ref_enc = ref_enc.unsqueeze(1)
        text_enc = self.text_encoder(text_input)

        attn = self.attention(ref_enc, text_enc, text_masks)
        attn = torch.softmax(attn, dim=1)

        feature_vec = torch.bmm(attn.unsqueeze(1), text_enc)
        feature_vec = feature_vec.squeeze(1)

        logits = self.linear_two(feature_vec)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.linear_one(logits)

        return logits, attn
