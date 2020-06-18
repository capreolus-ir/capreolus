from profane import import_all_modules

from .base import Task

import_all_modules(__file__, __package__)

from .rank import RankTask
from .rerank import RerankTask
