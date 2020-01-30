import os
import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.utils.common import padlist, get_default_cache_dir
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class EmbeddingHolder:
    """
    A utility class to load a pipeline and cache it in memory
    """

    PAD = "<pad>"
    SUPPORTED_EMBEDDINGS = {
        "glove6b": "glove/light/glove.6B.300d",
        "glove6b.50d": "glove/light/glove.6B.50d",
        "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
        "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
    }
    instances = {}

    def __init__(self, embedding_name):
        """
            If the _is_initialized class property is not set, build the benchmark and model (expensive)
            Else, do nothing.
        """
        self.embedding_name = embedding_name
        self.embedding = Magnitude(
            MagnitudeUtils.download_model(
                self.SUPPORTED_EMBEDDINGS[embedding_name], download_dir=os.environ.get("CAPREOLUS_CACHE", get_default_cache_dir())
            ),
            lazy_loading=-1,
            blocking=True,
        )
        self.stoi = {self.PAD: 0}  # string to integer. Associates an integer value with every token
        self.itos = {0: self.PAD}

    @classmethod
    def get_instance(cls, embedding_name):
        if not cls.instances.get(embedding_name):
            logger.debug("Caching embedding")
            cls.instances[embedding_name] = EmbeddingHolder(embedding_name)

        return cls.instances[embedding_name]

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def get_nvocab(self):
        # I have no idea what nvocab is. TODO: Figure this out - DSSM needs this
        return None

    def create_indexed_embedding_layer_from_tokens(self, tokens):
        """
        For each token in the list of tokens
        1. index = Converts the token into an integer
        2. embedding_for_token = Gets the embedding for the token from self.embedding
        3. creates a tensor where tensor[index] = embedding_for_token

        Why do we need to do this?
        We cannot use the downloaded magnitude embedding directly in a pytorch network. We need to convert into an
        indexed tensor and that is what we're doing here.
        :param tokens: A list of tokens
        :return: A tensor of dimension (len(tokens), self.embedding.dim)
        """
        tokens_minus_padding = [token for token in tokens if token != self.PAD]
        # Removing duplicates. Works only on python 3.7. See https://stackoverflow.com/a/7961390/1841522
        tokens_minus_padding = list(dict.fromkeys(tokens_minus_padding))
        vectors = self.embedding.query(tokens_minus_padding)
        indexed_embedding = np.zeros((len(vectors) + 1, self.embedding.dim), dtype=np.float32)
        indexed_embedding[self.stoi[self.PAD]] = np.zeros(self.embedding.dim)

        for i in range(0, len(vectors)):
            self.stoi[tokens_minus_padding[i]] = i + 1  # i + 1 because i starts from 0, and 0 is reserved for PAD
            self.itos[i + 1] = tokens_minus_padding[i]
            indexed_embedding[i + 1] = vectors[i]

        return indexed_embedding

    def get_index_array_from_tokens(self, tokens, maxlen):
        indices = [self.stoi.get(token, 0) for token in tokens]
        return np.array(padlist(indices, maxlen))
