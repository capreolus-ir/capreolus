import json

from capreolus import ModuleBase
from capreolus.utils.trec import load_qrels, load_trec_topics


class Benchmark(ModuleBase):
    """Base class for Benchmark modules. The purpose of a Benchmark is to provide the data needed to run an experiment, such as queries, folds, and relevance judgments.

    Modules should provide:
        - a ``topics`` dict mapping query ids (*qids*) to *queries*
        - a ``qrels`` dict mapping *qids* to *docids* and *relevance labels*
        - a ``folds`` dict mapping a fold name to *training*, *dev* (validation), and *testing* qids
        - if these can be loaded from files in standard formats, they can be specified by setting the ``topic_file``, ``qrel_file``, and ``fold_file``, respectively, rather than by setting the above attributes directly
    """

    module_type = "benchmark"
    qrel_file = None
    topic_file = None
    fold_file = None
    query_type = None
    relevance_level = 1
    """ Documents with a relevance label >= relevance_level will be considered relevant.
    This corresponds to trec_eval's --level_for_rel (and is passed to pytrec_eval as relevance_level). """

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
