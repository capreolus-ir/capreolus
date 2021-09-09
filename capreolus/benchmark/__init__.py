import os
import json
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from collections.abc import Iterable

import ir_datasets
import ir_measures
from ir_measures import *
from ir_measures.measures import Measure
from capreolus.searcher import Searcher

from capreolus import ModuleBase
from capreolus.utils.caching import cached_file, TargetFileExists
from capreolus.utils.trec import convert_metric, DEFAULT_METRICS, write_qrels, load_qrels, load_trec_topics
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


def validate(build_f):
    def validate_folds_file(self):
        if not hasattr(self, "fold_file"):
            logger.warning(f"Folds file is not found for Module {self.module_name}")
            return

        if self.fold_file.suffix != ".json":
            raise ValueError(f"Expect folds file to be in .json format.")

        raw_folds = json.load(open(self.fold_file))
        # we actually don't need to verify the name of folds right?

        for fold_name, fold_sets in raw_folds.items():
            if set(fold_sets) != {"train_qids", "predict"}:
                raise ValueError(f"Expect each fold to contain ['train_qids', 'predict'] fields.")

            if set(fold_sets["predict"]) != {"dev", "test"}:
                raise ValueError(f"Expect each fold to contain ['dev', 'test'] fields under 'predict'.")
        logger.info("Folds file validation finishes.")

    def validate_qrels_file(self):
        if not hasattr(self, "qrel_file"):
            logger.warning(f"Qrel file is not found for Module {self.module_name}")
            return

        n_dup, qrels = 0, defaultdict(dict)
        with open(self.qrel_file) as f:
            for line in f:
                qid, _, docid, label = line.strip().split()
                if docid in qrels[qid]:
                    n_dup += 1
                    if int(label) != qrels[qid][docid]:
                        raise ValueError(f"Found conflicting label in {self.qrel_file} for query {qid} and document {docid}.")
                qrels[qid][docid] = int(label)

        if n_dup > 0:
            qrel_file_no_ext, ext = os.path.splitext(self.qrel_file)
            dup_qrel_file = qrel_file_no_ext + "-contain-dup-entries" + ext
            os.rename(self.qrel_file, dup_qrel_file)
            write_qrels(qrels, self.qrel_file)
            logger.warning(
                f"Removed {n_dup} entries from the file {self.qrel_file}. The original version could be found in {dup_qrel_file}."
            )

        logger.info("Qrel file validation finishes.")

    def validate_query_alignment(self):
        topic_qids = set(self.topics[self.query_type])
        qrels_qids = set(self.qrels)

        for fold_name, fold_sets in self.folds.items():
            # check if there are overlap between training, dev, and test set
            train_qids, dev_qids, test_qids = (
                set(fold_sets["train_qids"]),
                set(fold_sets["predict"]["dev"]),
                set(fold_sets["predict"]["test"]),
            )
            if len(train_qids & dev_qids) > 0:
                logger.warning(
                    f"Found {len(train_qids & dev_qids)} overlap queries between training and dev set in fold {fold_name}."
                )
            if len(train_qids & test_qids) > 0:
                logger.warning(
                    f"Found {len(train_qids & dev_qids)} overlap queries between training and dev set in fold {fold_name}."
                )
            if len(dev_qids & test_qids) > 0:
                logger.warning(
                    f"Found {len(train_qids & dev_qids)} overlap queries between training and dev set in fold {fold_name}."
                )

            # check if the topics, qrels, and folds file share a reasonable set (if not all) of queries
            folds_qids = train_qids | dev_qids | test_qids
            n_overlap = len(set(topic_qids) & set(qrels_qids) & set(folds_qids))
            if not len(topic_qids) == len(qrels_qids) == len(folds_qids) == n_overlap:
                logger.warning(
                    f"Number of queries are not aligned across topics, qrels and folds in fold {fold_name}: {len(topic_qids)} queries in topics file, {len(qrels_qids)} queries in qrels file, {len(folds_qids)} queries in folds file; {n_overlap} overlap queries found among the three."
                )

            # check if any topic in folds cannot be found in topics file
            for set_name, set_qids in zip(["training", "dev", "test"], [train_qids, dev_qids, test_qids]):
                if len(set_qids - topic_qids) > 0:
                    raise ValueError(
                        f"{len(set_qids - topic_qids)} queries in {set_name} set of fold {fold_name} cannot be found in topic file."
                    )

        logger.info("Query Alignment validation finishes.")

    def _validate(self):
        """Rewrite the files that contain invalid (duplicate) entries, and remove the currently loaded variables"""
        build_f(self)
        validate_folds_file(self)
        validate_qrels_file(self)
        validate_query_alignment(self)

    return _validate


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
    use_train_as_dev = True
    """ Whether to use training set as validate set when there is no training needed,
    e.g. for traditional IR algorithms like BM25 """

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

    @property
    def non_nn_dev(self):
        dev_per_fold = {fold_name: deepcopy(folds["predict"]["dev"]) for fold_name, folds in self.folds.items()}
        if self.use_train_as_dev:
            for fold_name, folds in self.folds.items():
                dev_per_fold[fold_name].extend(folds["train_qids"])
        return dev_per_fold

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

    @validate
    def build(self):
        return

    def evaluate(self, runs_or_runfile, qrels=None, metrics=None):
        """Evaluate a runs dictionary or runfile. The benchmark.relevance_level would be applied silently in this function."""
        if qrels is None:
            qrels = self.qrels

        if metrics is None:
            metrics = DEFAULT_METRICS
        metrics = list(map(convert_metric, metrics))
        assert all(isinstance(m, Measure) for m in metrics)

        metrics_rel2ori = {}
        for metric in metrics:
            rel_metric = metric(rel=self.relevance_level) if "rel" in metric.SUPPORTED_PARAMS else metric
            metrics_rel2ori[rel_metric] = metric
        assert set(metrics_rel2ori.values()) == set(metrics)

        # prepare runs dictionary
        if isinstance(runs_or_runfile, Path) or isinstance(runs_or_runfile, str):
            if not os.path.exists(runs_or_runfile):
                raise ValueError(f"Cannot find run file {runs_or_runfile}")
            runs = ir_measures.read_trec_run(runs_or_runfile)
        else:
            runs = runs_or_runfile

        try:
            scores = ir_measures.calc_aggregate(list(metrics_rel2ori.keys()), qrels, runs)
        except OSError as e:
            logger.warning(e)
            scores = {k: -1 for k in metrics_rel2ori}

        scores = {metrics_rel2ori[k]: v for k, v in scores.items()}
        return scores

    def search_best_run(self, runfile_dirs, primary_metric, metrics=DEFAULT_METRICS, folds=None):
        if not isinstance(runfile_dirs, (list, tuple)):
            assert isinstance(runfile_dirs, Path)
            runfile_dirs = [runfile_dirs]

        assert isinstance(metrics, Iterable)
        metrics = list(metrics) if not isinstance(metrics, list) else metrics
        if primary_metric not in metrics:
            metrics = [primary_metric] + metrics

        folds = {s: self.folds[s] for s in [folds]} if folds else self.folds
        runfiles = [
            os.path.join(runfile_dir, f)
            for runfile_dir in runfile_dirs
            for f in os.listdir(runfile_dir)
            if (f != "done" and not os.path.isdir(os.path.join(runfile_dir, f)))
        ]

        best_scores = {s: {primary_metric: 0, "path": None} for s in folds}
        for runfile in runfiles:
            for fold_name in folds:
                dev_qrels = {qid: self.qrels[qid] for qid in self.non_nn_dev[fold_name] if qid in self.qrels}
                runs = ir_measures.read_trec_run(runfile)
                score = self.evaluate(runs, dev_qrels, [primary_metric])[primary_metric]
                if score > best_scores[fold_name][primary_metric]:
                    best_scores[fold_name] = {primary_metric: score, "path": runfile}

        for fold, scores in best_scores.items():
            logger.info(f"Best dev score on fold {fold}: {primary_metric}={scores[primary_metric]}")

        test_runs = {}
        for s, score_dict in best_scores.items():
            test_qids = folds[s]["predict"]["test"]
            # any empty (no results) queries need to be added so they contribute zeros to the average
            test_runs.update({qid: {} for qid in test_qids})
            test_runs.update({qid: v for qid, v in Searcher.load_trec_run(score_dict["path"]).items() if qid in test_qids})

        scores = self.evaluate(test_runs, metrics=metrics)
        return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}


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
                qrels.setdefault(qrel.query_id, {})
                qrels[qrel.query_id][qrel.doc_id] = max(qrel.relevance, qrels[qrel.query_id].get(qrel.doc_id, -1))

        return qrels

    def ird_load_topics(self):
        topics = {}
        field = "description" if self.query_type == "desc" else self.query_type

        for name in self.ird_dataset_names:
            dataset = ir_datasets.load(name)
            for query in dataset.queries_iter():
                topics[query.query_id] = getattr(query, field).replace("\n", " ")

        return {self.query_type: topics}


from profane import import_all_modules

from .dummy import DummyBenchmark

import_all_modules(__file__, __package__)
