from capreolus import ModuleBase


class Tokenizer(ModuleBase):
    """Base class for Tokenizer modules. The purpose of a Tokenizer is to tokenize strings of text (e.g., as required by an :class:`~capreolus.extractor.Extractor`).

    Modules should provide:
        - a ``tokenize(strings)`` method that takes a list of strings and returns tokenized versions
    """

    module_type = "tokenizer"


from profane import import_all_modules

from .anserini import AnseriniTokenizer
from .bert import BertTokenizer

import_all_modules(__file__, __package__)
