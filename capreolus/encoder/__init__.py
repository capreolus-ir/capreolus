from capreolus import Dependency, ModuleBase, get_logger

logger = get_logger(__name__)


class Encoder(ModuleBase):
    """
    Base class for encoders. Encoders take a document and convert it into a vector, and the result is usually put in a FAISS index for an approximate nearest-neighbour search
    """

    module_type = "encoder"
    def build_model(self):
        """
        Initialize the PyTorch model
        """
        raise NotImplementedError

    def encode(self, numericalized_doc_toks):
        """
        Accepts a document (or a list of documents) and returns the corresponding vectors
        """
        # 1. Numericalize the document. self.tokenizer.tokenize(document) should be enough
        # 2. self.model(numericalized_document)
        
        raise NotImplementedError


from profane import import_all_modules

from .TinyBERT import TinyBERTEncoder


import_all_modules(__file__, __package__)
