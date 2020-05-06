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
    domain = 'book'

    @staticmethod
    def config():
        querytype = "query"  # one of: query, basicprofile, entityprofile
        entity_strategy = 'none'

        if querytype not in ["query", "basicprofile", "entityprofile"]:
            raise ValueError(f"invalid querytype: {querytype}")

        if entity_strategy not in ['none', 'all', 'domain']: #TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        if querytype == 'entityprofile' and entity_strategy != 'none':
            raise ValueError(f"wrong usage of incorporate entities. We cannot use it with querytype 'entityprofile'")

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
    def incorporate_entities(self):
        return False if self.cfg["entity_strategy"] == 'none' else True

    @property
    def entity_strategy(self):
        return self.cfg['entity_strategy']

    @property
    def domain(self):
        return self.domain

    @property
    def topic_file(self):
        fn = f"topics.{self.query_type}.txt"
        return self.PES20_DIR / fn

class KITT(Benchmark):
    name = "kitt"
    DATA_DIR = Path("/GW/PKB/work/data_personalization/TREC_format/")
    # DATA_DIR = Path("/home/ghazaleh/workspace/capreolus/data/test/")
    qrel_file = DATA_DIR / "judgements"
    fold_file = DATA_DIR / "splits.json"

    @staticmethod
    def config():
        querytype = "query"
        domain = "book"
        entity_strategy = 'none' ##TODO: I don't like that this is a string, and will be used in other modules... how can this be handled better though?

        if querytype not in ["query", "basicprofile", "chatprofile",
                             "basicprofile_general", 'basicprofile_food', 'basicprofile_travel', 'basicprofile_book_movie',
                             "chatprofile_general", 'chatprofile_food', 'chatprofile_travel', 'chatprofile_book', 'chatprofile_movie', 'chatprofile_hobbies']:
            raise ValueError(f"invalid querytype: {querytype}")

        if domain not in ["book", "travel_wikivoyage", "movie", "food"]:
            raise ValueError(f"invalid domain: {domain}")

        if entity_strategy not in ['none', 'all', 'domain']: #TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        KITT.qrel_file = KITT.DATA_DIR / "{}_judgements".format(domain)
        KITT.fold_file = KITT.DATA_DIR / "{}_splits.json".format(domain)

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

    # @property
    # def incorporate_entities(self):
    #     return False if self.cfg["entity_strategy"] == 'none' else True

    @property
    def domain(self):
        return self.cfg["domain"]

    @property
    def entity_strategy(self):
        return self.cfg['entity_strategy']

    @property
    def topic_file(self):
        fn = f"{self.domain}_topics.{self.query_type}.txt"
        return self.DATA_DIR / fn

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
