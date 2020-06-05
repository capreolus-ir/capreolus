import multiprocessing
import os
from pathlib import Path

from profane import constants

# specify a base package that we should look for modules under (e.g., <BASE>.task)
# constants must be specified before importing Task (or any other modules!)
constants["BASE_PACKAGE"] = "capreolus"
# specify other constants that modules expect
constants["PACKAGE_PATH"] = Path(os.path.dirname(__file__))
constants["CACHE_BASE_PATH"] = Path(os.environ.get("CAPREOLUS_CACHE", os.path.expanduser("~/.capreolus/cache/")))
constants["RESULTS_BASE_PATH"] = Path(os.environ.get("CAPREOLUS_RESULTS", os.path.expanduser("~/.capreolus/results/")))
constants["MAX_THREADS"] = int(os.environ.get("CAPREOLUS_THREADS", multiprocessing.cpu_count()))

import jnius_config
from capreolus.utils.common import Anserini

jnius_config.set_classpath(Anserini.get_fat_jar())
