from capreolus import Dependency, ModuleBase, get_logger

logger = get_logger(__name__)


class Encoder(ModuleBase):
    """
    Base class for encoders. Encoders take a document and convert it into a vector, and the result is usually put in a FAISS index for an approximate nearest-neighbour search
    """

    module_type = "encoder"
    dependencies = [Dependency(key="tokenizer", module="tokenizer")]

    def build(self):
        """
        Initialize the PyTorch model
        """
        raise NotImplementedError

    def encode(document):
        """
        Accepts a document (or a list of documents) and returns the corresponding vectors
        """
        # 1. Numericalize the document. self.tokenizer.tokenize(document) should be enough
        # 2. self.model(numericalized_document)
        
        raise NotImplementedError


        
