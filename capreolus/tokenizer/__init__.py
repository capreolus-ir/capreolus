from capreolus import ModuleBase, Dependency, ConfigOption


class Tokenizer(ModuleBase):
    module_type = "tokenizer"


from profane import import_all_modules

from .anserini import AnseriniTokenizer
from .bert import BertTokenizer

import_all_modules(__file__, __package__)
