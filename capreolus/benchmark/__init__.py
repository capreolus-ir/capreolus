import json
from os.path import join
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
    entity_strategy = None

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
    PES20_DIR = Path(open(join(PACKAGE_PATH, "..", "paths_env_vars", "pes20benchmarkpath"), 'r').read().strip())
    qrel_file = PES20_DIR / "judgements"
    fold_file = PES20_DIR / "splits.json"
    domain = 'book'

    @staticmethod
    def config():
        querytype = "query"  # one of: query, basicprofile, entityprofile

        if querytype not in ["query", "basicprofile", "entityprofile",
                             'basicprofile_demog_hobbies', 'basicprofile_minus_books',
                             'entityprofile_demog_hobbies', 'entityprofile_minus_books']:
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
    def incorporate_entities(self):
        return False if self.cfg["entity_strategy"] is None else True

    @property
    def topic_file(self):
        fn = f"topics.{self.query_type}.txt"
        return self.PES20_DIR / fn

class KITT(Benchmark):
    name = "kitt"
    DATA_DIR = Path(open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip())

    @staticmethod
    def config():
        querytype = "query"
        domain = "book"
        assessed_set = 'random20'
        if querytype not in ['query', 'basicprofile', 'chatprofile',
                              'basicprofile_general', 'basicprofile_food', 'basicprofile_travel', 'basicprofile_book',
                              'basicprofile_food_general', 'basicprofile_travel_general', 'basicprofile_book_general',
                              'chatprofile_general', 'chatprofile_food', 'chatprofile_travel', 'chatprofile_book',
                              'chatprofile_food_general', 'chatprofile_travel_general', 'chatprofile_book_general'
                             ]:
            raise ValueError(f"invalid querytype: {querytype}")

        if domain not in ["book", "travel", "food", "alldomains"]:
            raise ValueError(f"invalid domain: {domain}")

        if assessed_set not in ['all', 'random20', 'top10']:
            raise ValueError(f"invalid assessed_set: {assessed_set}")

        KITT.qrel_file = KITT.DATA_DIR / "judgements" / "{}_judgements_{}".format(domain, assessed_set)
        KITT.fold_file = KITT.DATA_DIR / "splits" / "{}_folds.json".format(domain)

    @property
    def topic_file(self):
        fn = f"{self.domain}_topics.{self.query_type}.txt"
        return self.DATA_DIR / "topics" / fn

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
    def domain(self):
        return self.cfg["domain"]


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
