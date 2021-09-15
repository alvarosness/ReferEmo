from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class BaseTokenizer:
    """Base tokenizer class. All tokenizers inherit from this base class"""

    def tokenize(self, corpus: List[str]):
        raise NotImplementedError()

    def __call__(self, corpus: List[str]):
        return self.tokenize(corpus)


class BERTTokenizer(BaseTokenizer):
    """BERT specific text tokenizer"""

    def __init__(self, pretrained_path: str = 'bert-base-uncased', padding: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.padding = padding

    def tokenize(self, corpus: List[str]):
        return self.tokenizer(corpus, padding=self.padding, return_tensors='pt')


class BiLSTMTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.padding = True
        self.vocab = None

    def tokenize(self, corpus: List[str]):
        vectors = [self._vectorize_document(doc) for doc in corpus]

        if self.padding:
            return pad_sequence(vectors, padding_value=self.vocab['PAD'], batch_first=True)

        return vectors

    def _vectorize_document(self, doc: str):
        vector = [self.vocab[word] for word in doc.split(' ')]

        return torch.LongTensor(vector)
