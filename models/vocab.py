import os
import pickle
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import requests
from tqdm import tqdm


class Vocab:
    """Basic vocabulary object containing the mapping of tokens to indices and vice versa."""

    def __init__(self, filepath: str = None) -> None:
        self.word2index = {'PAD': 0, 'UNK': 1}
        self.index2word = ['PAD', 'UNK']

        if filepath:
            self.load(filepath)

    def __len__(self):
        return len(self.index2word)

    def __repr__(self):
        return self.__dict__

    def __str__(self):
        return f"<Vocab with size: {len(self)}>"

    def __getitem__(self, item):
        return self.word2index.get(item, self.word2index["UNK"])

    def add_sentence(self, sentence: str):
        """Adds each word in the sentence to the vocabulary. Assumes words are separated by spaces."""
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word: str):
        """Add word to vocabulary if it doesn't already exist."""
        if word in self.word2index:
            return

        self.word2index[word] = len(self.index2word)
        self.index2word.append(word)

    def save(self, filepath: str):
        """Saves the vocabulary to the specified file. Data is saved as a binary object."""
        with open(filepath, 'wb') as fp:
            pickle.dump(self.__dict__, fp)

    def load(self, filepath: str):
        """Loads the vocabulary from the specified file. Assumes a binary file."""
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
            self.word2index = data['word2index']
            self.index2word = data['index2word']


class GloVe:
    """Object storing the word vectors and a vocabulary object for the words."""
    URLS = {
        "glove.6B": "https://nlp.stanford.edu/data/glove.6B.zip",
        "glove.42B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "glove.840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "glove.twitter.27B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    }

    DIMS = {
        "glove.6B": [50, 100, 200, 300],
        "glove.42B": [300],
        "glove.840B": [300],
        "glove.twitter.27B": [25, 50, 100, 200]
    }

    def __init__(self, filepath=None, size: str = "6B", dim: int = 300, cache: str = ".vector_cache") -> None:
        """
        Parameters
        ---
        filepath - path to a file containing word vectors. (default = None)
        size - size of the GloVe word vectors. Possible values are 6B, 42B, 840B, and twitter.27B. (default = 6B)
        dim - dimension of the GloVe word vectors. 
        cache - cache directory to store the downloaded word vectors
        """

        self.vocab = Vocab()
        self.vectors = None

        if filepath:
            self._load_from_file(filepath)
        else:
            self._download_and_extract(size, dim, cache)

    def __getitem__(self, item):
        return self.vectors[self.vocab[item]]

    def save(self, filepath: str):
        """
        Saves the word vectors in a file.
        Each row has the word at the first column and the vector value in the remaining columns
        """
        with open(filepath, 'w') as fp:
            for idx in range(2, len(self.vocab)):
                word = self.vocab.index2word[idx]
                vec_str = " ".join([str(num) for num in self.vectors[idx]])
                vec_str = word + " " + vec_str + "\n"

                fp.write(vec_str)

    def _download_and_extract(self, size: str, dim: int, cache: str):
        """
        Downloads GloVe word vectors if they aren't downloaded to the cache already and then loads said word vectors.
        """

        glove_size = f"glove.{size}"

        glove_filename = f"{glove_size}.{dim}d.txt"
        glove_filename = os.path.join(cache, glove_filename)

        if glove_size not in self.URLS:
            raise ValueError(
                "Selected GloVe configuration is not implemented.")

        if dim not in self.DIMS[glove_size]:
            raise ValueError("Selected dimension is not available.")

        # Check if the file doesn't already exist
        if not os.path.exists(glove_filename):
            self._download_vectors(self.URLS[glove_size], cache)
        self._load_from_file(glove_filename)

    def _download_vectors(self, url: str, cache: str):
        """Downloads word vector zip files and extracts them to the cache directory."""
        chunk_size = 1024 * 1024

        r = requests.get(url, stream=True)
        total_size = int(r.headers['content-length'])

        with NamedTemporaryFile() as fp:
            for data in tqdm(iterable=r.iter_content(chunk_size=chunk_size), total=total_size/chunk_size, unit='KB'):
                fp.write(data)

            # Rewinding the file pointer to guarantee that we are
            # at the start of the file when extracting
            fp.seek(0)
            with ZipFile(fp.name) as zip:
                zip.extractall(cache)

    def _load_from_file(self, filepath: str):
        """Loads word vectors from file."""
        with open(filepath) as fp:
            vecs = []

            for line in fp:
                values = line.split(' ')
                word = values[0]
                vec = [float(num) for num in values[1:]]

                self.vocab.add_word(word)
                vecs.append(vec)

        self.vectors = np.array(vecs, dtype=np.float32)

        pad_vec = np.zeros_like(self.vectors[0])
        # Reshaping so that vector becomes 1xdim
        pad_vec = np.reshape(pad_vec, (1, -1))

        unk_vec = np.mean(self.vectors, axis=0)
        # Reshaping so that vector becomes 1xdim
        unk_vec = np.reshape(unk_vec, (1, -1))

        self.vectors = np.concatenate((pad_vec, unk_vec, self.vectors))
