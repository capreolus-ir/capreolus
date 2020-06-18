from profane import import_all_modules

from .base import Benchmark
from .dummy import DummyBenchmark

import_all_modules(__file__, __package__)
