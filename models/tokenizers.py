from typing import List
from transformers import AutoTokenizer


class BaseTokenizer:
    """Base tokenizer class. All tokenizers inherit from this base class"""

    def tokenize(self, text: List[str]):
        pass

    def __call__(self, text):
        return self.tokenize(text)


class BERTTokenizer(BaseTokenizer):
    """BERT specific text tokenizer"""

    def __init__(self, pretrained_path: str = 'bert-base-uncased', padding: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.padding = padding

    def tokenize(self, text: List[str]):
        return self.tokenizer(text, padding=self.padding, return_tensors='pt')
