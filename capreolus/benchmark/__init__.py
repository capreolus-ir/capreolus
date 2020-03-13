import json
import os
from pathlib import Path

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


class PES20(Benchmark):
    name = "pes20"
    PES20_DIR = Path("/GW/NeuralIR/work/PES20")  # TODO hardcoded path
    qrel_file = PES20_DIR / "judgements"
    fold_file = PES20_DIR / "splits.json"

    @staticmethod
    def config():
        querytype = "query"  # one of: query, basicprofile, entityprofile

        if querytype not in ["query", "basicprofile", "entityprofile"]:
            raise ValueError(f"invalid querytype: {querytype}")

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.topic_file)
            assert self.query_type not in self._topics
            self._topics[self.query_type] = self._topics["title"]
        return self._topics

    @property
    def query_type(self):
        return self.cfg["querytype"]

    @property
    def topic_file(self):
        fn = f"topics.{self.query_type}.txt"
        return self.PES20_DIR / fn


class DummyBenchmark(Benchmark):
    name = "dummy"
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"


class WSDM20Demo(Benchmark):
    name = "wsdm20demo"
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


class Robust04Yang19(Benchmark):
    name = "robust04.yang19"
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


class ANTIQUE(Benchmark):
    name = "antique"
    qrel_file = PACKAGE_PATH / "data" / "qrels.antique.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.antique.txt"
    fold_file = PACKAGE_PATH / "data" / "antique.json"
    query_type = "title"


class MSMarcoPassage(Benchmark):
    name = "msmarcopassage"
    qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcopassage.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.msmarcopassage.txt"
    fold_file = PACKAGE_PATH / "data" / "msmarcopassage.folds.json"
    query_type = "title"
