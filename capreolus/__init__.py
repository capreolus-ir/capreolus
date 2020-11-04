import multiprocessing
import os
from pathlib import Path

from profane import ConfigOption, Dependency, ModuleBase, constants, config_list_to_dict, module_registry


__version__ = "0.2.5"


### set constants used by capreolus and profane ###
# these must be specified before importing Task (or any other modules!)
#
# specify a base package that profane should look for modules under (e.g., <BASE>.task)
constants["BASE_PACKAGE"] = "capreolus"
# specify constants that capreolus modules expect
constants["PACKAGE_PATH"] = Path(os.path.dirname(__file__))
constants["CACHE_BASE_PATH"] = Path(os.environ.get("CAPREOLUS_CACHE", os.path.expanduser("~/.capreolus/cache/")))
constants["RESULTS_BASE_PATH"] = Path(os.environ.get("CAPREOLUS_RESULTS", os.path.expanduser("~/.capreolus/results/")))
constants["MAX_THREADS"] = int(os.environ.get("CAPREOLUS_THREADS", multiprocessing.cpu_count()))


### set missing environment variables to safe defaults ###
if "GENSIM_DATA_DIR" not in os.environ:
    os.environ["GENSIM_DATA_DIR"] = (constants["CACHE_BASE_PATH"] / "gensim").as_posix()

if "NLTK_DATA" not in os.environ:
    os.environ["NLTK_DATA"] = (constants["CACHE_BASE_PATH"] / "nltk").as_posix()

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


import jnius_config
from capreolus.utils.common import Anserini

jnius_config.set_classpath(Anserini.get_fat_jar())


### convenience imports
# note: order is important to avoid circular imports
from capreolus.utils.loginit import get_logger
from capreolus.benchmark import Benchmark
from capreolus.collection import Collection
from capreolus.index import Index
from capreolus.searcher import Searcher
from capreolus.extractor import Extractor
from capreolus.reranker import Reranker
from capreolus.tokenizer import Tokenizer
from capreolus.trainer import Trainer
from capreolus.task import Task


def parse_config_string(s):
    """ Convert a config string to a config dict """
    s = " ".join(s.split())  # remove consecutive whitespace
    return config_list_to_dict(s.split())
