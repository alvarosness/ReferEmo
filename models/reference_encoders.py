"""Module containing all of the reference encoders used in the architecture
"""
from torch import nn
from transformers import AutoModel


class BERTReferenceEncoder(nn.Module):
    """
    BERT-based reference encoder.

    This encoder generates a feature vector that serves to enrich token embeddings with emotional information.
    """

    def __init__(self, pretrained_path: str = 'bert-base-uncased'):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_path)
        self.output_size = self.bert.config.hidden_size

    def forward(self, inputs):
        outputs = self.bert(**inputs, output_attentions=True)

        return outputs.pooler_output, outputs.attentions
