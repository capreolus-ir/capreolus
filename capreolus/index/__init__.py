from profane import import_all_modules

from .base import Index
from .anserini import AnseriniIndex

import_all_modules(__file__, __package__)
