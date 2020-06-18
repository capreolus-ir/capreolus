from profane import import_all_modules

from .base import Tokenizer
from .anserini import AnseriniTokenizer
from .bert import BertTokenizer

import_all_modules(__file__, __package__)
