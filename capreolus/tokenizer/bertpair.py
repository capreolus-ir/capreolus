import os
import sys

from pytorch_transformers import BertTokenizer

from capreolus.utils.common import register_component_module, import_component_modules, args_to_key, get_default_cache_dir
from capreolus.tokenizer import Tokenizer
from capreolus.utils.cache_capnp import Document
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Tokenizer.register
class BertPairTokenizer(Tokenizer):
    name = "bertpair"

    def __init__(self, index, tokmodel="bert-base-uncased", use_cache=True):
        super().__init__(index)
        self.tokmodel = tokmodel
        self.params = {"tokmodel": self.tokmodel.replace("-", "")}

        if use_cache:
            self.initialize_cache(self.params)

    def create(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.tokmodel, cache_dir=os.environ.get("CAPREOLUS_CACHE", get_default_cache_dir())
        )
        self.vocab = self.tokenizer.vocab

    def tokenize(self, s):
        return self.tokenizer.encode(s)


import_component_modules("tokenizer")
