import json
import os

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.trec import load_qrels, load_trec_topics


class Benchmark(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "benchmark"
    qrel_file = None
    topic_file = None
    fold_file = None
    query_type = None

    @property
    def qrels(self):
        if not hasattr(self, "_qrels"):
            self._qrels = load_qrels(self.qrel_file)
        return self._qrels

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.topic_file)
        return self._topics

    @property
    def folds(self):
        if not hasattr(self, "_folds"):
            self._folds = json.load(open(self.fold_file, "rt"))
        return self._folds


class DummyBenchmark(Benchmark):
    name = "dummy"
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"

    @staticmethod
    def config():
        fold = "s1"
        rundocsonly = False


class WSDM20Demo(Benchmark):
    name = "wsdm20demo"
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"

    @staticmethod
    def config():
        fold = "s1"
        rundocsonly = True  # use only docs from the searcher as pos/neg training instances (i.e., not all qrels)