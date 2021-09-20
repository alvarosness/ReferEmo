from dataclasses import dataclass
from typing import Any, List, Union

import spacy
import torch
from torch.utils.data import Dataset

EMOTICON_TO_EMOJI = {
    ':-)': "🙂", ':)': "🙂", ':-]': "🙂", ':]': "🙂",
    '=)': "🙂", '=]': "🙂", ':-3': "😊", ':3': "😊",
    '=3': "😊", ':-D': "😃", ':D': "😃", 'XD': "😆",
    'X-D': "😆", ':-O': "😮", ':O': "😮", ':-o': "😮",
    ':o': "😮", ':(': "🙁", ':-(': "🙁", ':P': "😛",
    ':-P': "😛", ':^(': "🙁"
}


@dataclass
class ReferEmoModelInput:
    """ReferEmo model input"""
    reference_inputs: Any
    text_inputs: Any
    text_masks: Any

    def to(self, device=torch.device('cpu')):
        self.reference_inputs = {key: value.to(
            device) for key, value in self.reference_inputs.items()}
        self.text_inputs = self.text_inputs.to(device)
        self.text_masks = self.text_masks.to(device)

        return self


class EmotionDataset(Dataset):
    def __init__(self, data, labels, label_names) -> None:
        super().__init__()

        self.data = data
        self.labels = labels
        self.label_names = label_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class DataProcessor:
    """Class responsible for generating the data in the format accepted by the model"""
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, text_tokenizer, ref_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.ref_tokenizer = ref_tokenizer

    def _replace_emoticon_with_emoji(self, corpus):
        for idx in range(len(corpus)):
            for emoticon, emoji in EMOTICON_TO_EMOJI.items():
                corpus[idx] = corpus[idx].replace(emoticon, emoji)

        return corpus

    def clean_text(self, corpus: List[str]):
        cleaned_corpus = []

        for doc in self.nlp.pipe(corpus):
            cleaned_doc = " ".join([token.text.lower()
                                    for token in doc if not token.is_space and not token.like_url])
            cleaned_corpus.append(cleaned_doc)

        return cleaned_corpus

    def generate_batch_input(self, data, labels=None):
        # Clean text (e.g., remove links and convert emoticons to emojis)
        data = self._replace_emoticon_with_emoji(data)
        data = self.clean_text(data)

        # Use the text tokenizer to generate the input tokens
        text_tokens, masks = self.text_tokenizer(data)
        ref_tokens = self.ref_tokenizer(data)

        # Generate ModelInput list
        inputs = ReferEmoModelInput(ref_tokens, text_tokens, masks)

        # Torchify labels if they exist
        if labels is not None:
            labels = labels.to(dtype=torch.float64)
            return inputs, labels

        return inputs
