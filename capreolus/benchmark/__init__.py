import json

from capreolus import ModuleBase
from capreolus.utils.trec import load_qrels, load_trec_topics


class Benchmark(ModuleBase):
    """the module base class"""

    module_type = "benchmark"
    qrel_file = None
    topic_file = None
    fold_file = None
    query_type = None
    # documents with a relevance label >= relevance_level will be considered relevant
    # corresponds to trec_eval's --level_for_rel (and passed to pytrec_eval as relevance_level)
    relevance_level = 1

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


from profane import import_all_modules

from .dummy import DummyBenchmark

import_all_modules(__file__, __package__)
