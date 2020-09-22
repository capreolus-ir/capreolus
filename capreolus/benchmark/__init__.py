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
    PES20_DIR = Path("/GW/NeuralIR/work/PES20")  # TODO hardcoded path
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
    DATA_DIR = Path("/GW/PKB/work/data_personalization/TREC_format_quselection_C_final_profiles/")
    qrel_file = DATA_DIR / "judgements"
    fold_file = DATA_DIR / "splits.json"

    @staticmethod
    def config():
        querytype = "query"
        domain = "book"
        assessed_set = None
#TODO make decision on this... with hobbies or without, it effects the baseprofile choosing
        if querytype not in ['query', 'basicprofile', 'chatprofile', 'basicprofileMR', 'chatprofileMR',
                              'basicprofile_general',
                              'basicprofile_food', 'basicprofile_travel',
                              'basicprofile_book', 'basicprofile_movie',
                              'basicprofile_food_general', 'basicprofile_travel_general',
                              'basicprofile_book_general', 'basicprofile_movie_general',
                              'chatprofile_general',
                              'chatprofile_food', 'chatprofile_travel',
                              'chatprofile_book', 'chatprofile_movie',
                              'chatprofile_food_general', 'chatprofile_travel_general',
                              'chatprofile_book_general', 'chatprofile_movie_general',
                             ]:
            raise ValueError(f"invalid querytype: {querytype}")

        if domain not in ["book", "travel_wikivoyage", "movie", "food", "alldomains", "alldomainsMR"]:
            raise ValueError(f"invalid domain: {domain}")

        if assessed_set not in [None, 'random20', 'top10']:
            raise ValueError(f"invalid assessed_set: {assessed_set}")

        KITT.qrel_file = KITT.DATA_DIR / "{}_judgements".format(domain)
        KITT.fold_file = KITT.DATA_DIR / "{}_splits.json".format(domain)
        if assessed_set is not None:
            KITT.qrel_file = KITT.DATA_DIR / "{}_judgements_{}".format(domain, assessed_set)

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
