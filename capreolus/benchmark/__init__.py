import json
import os

import ir_datasets

from capreolus import ModuleBase
from capreolus.utils.caching import cached_file, TargetFileExists
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
            self._folds = json.load(open(self.fold_file, "rt"), parse_int=str)
        return self._folds

    def get_topics_file(self, query_sets=None):
        """Returns path to a topics file in TSV format containing queries from query_sets.
        query_sets may contain any combination of 'train', 'dev', and 'test'.
        All are returned if query_sets is None."""

        if query_sets:
            query_sets = set(query_sets)
            invalid = query_sets - {"train", "test", "dev"}
            if invalid:
                raise ValueError(f"query_sets contains invalid fold names: {invalid}")
            query_sets = "_".join(sorted(query_sets))

            valid_qids = set()
            if "train" in query_sets:
                valid_qids.update(self.folds["train_qids"])
            if "dev" in query_sets:
                valid_qids.update(self.folds["predict"]["dev"])
            if "test" in query_sets:
                valid_qids.update(self.folds["predict"]["test"])
        else:
            query_sets = "all"
            valid_qids = None

        fn = self.get_cache_path() / f"topics-{query_sets}.tsv"

        try:
            with cached_file(fn) as tmp_fn:
                with open(tmp_fn, "wt") as outf:
                    for qid, query in self.topics[self.query_type].items():
                        if query_sets == "all" or qid in valid_qids:
                            print(f"{qid}\t{query}", file=outf)
        except TargetFileExists as e:
            pass

        return fn


class IRDBenchmark(Benchmark):
    ird_dataset_names = []

    @property
    def qrels(self):
        if not hasattr(self, "_qrels"):
            self._qrels = self.ird_load_qrels()
        return self._qrels

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = self.ird_load_topics()
        return self._topics

    def ird_load_qrels(self):
        qrels = {}
        for name in self.ird_dataset_names:
            dataset = ir_datasets.load(name)
            for qrel in dataset.qrels_iter():
                qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return qrels

    def ird_load_topics(self):
        topics = {}
        field = "description" if self.query_type == "desc" else self.query_type

        for name in self.ird_dataset_names:
            dataset = ir_datasets.load(name)
            for query in dataset.queries_iter():
                topics[query.query_id] = getattr(query, field)

        return {field: topics}


from profane import import_all_modules

from .dummy import DummyBenchmark

import_all_modules(__file__, __package__)
