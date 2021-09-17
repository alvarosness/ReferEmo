from typing import List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from models.vocab import Vocab


class BaseTokenizer:
    """Base tokenizer class. All tokenizers inherit from this base class"""

    def tokenize(self, corpus: Union[List[str], str]):
        raise NotImplementedError()

    def __call__(self, corpus: Union[List[str], str]):
        return self.tokenize(corpus)


class BERTTokenizer(BaseTokenizer):
    """BERT specific text tokenizer"""

    def __init__(self, pretrained_path: str = 'bert-base-uncased', padding: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.padding = padding

    def tokenize(self, corpus: Union[List[str], str]):
        return self.tokenizer(corpus, padding=self.padding, return_tensors='pt')


class BiLSTMTokenizer(BaseTokenizer):
    def __init__(self, vocab: Vocab):
        super().__init__()
        self.padding = True
        self.vocab = vocab

    def tokenize(self, corpus: Union[List[str], str]):
        if isinstance(corpus, str):
            corpus = [corpus]

        vectors = [self._vectorize_document(doc) for doc in corpus]
        seq_lens = [len(doc) for doc in vectors]

        if self.padding:
            max_len = max(seq_lens)
            seq_lens = torch.LongTensor(seq_lens)
            mask = torch.arange(max_len)[None, :] < seq_lens[:, None]

            return pad_sequence(vectors, padding_value=self.vocab['PAD'], batch_first=True), mask

        return vectors, seq_lens

    def _vectorize_document(self, doc: str):
        vector = [self.vocab[word] for word in doc.split(' ')]

        return torch.LongTensor(vector)
