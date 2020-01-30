import json
import os

from collections import defaultdict

import numpy as np
import pytrec_eval

from capreolus.utils.common import register_component_module, import_component_modules
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Searcher:
    """ Module responsible for searching an Index. Searchers are usually coupled to an Index module (e.g., AnseriniIndex). """
    ALL = {}

    def __init__(self, index, collection, run_path, pipe_config):
        self.index = index
        self.collection = collection
        self.run_path = run_path
        self.pipeline_config = pipe_config

    @staticmethod
    def config():
        return locals().copy()  # ignored by sacred

    @classmethod
    def register(cls, subcls):
        return register_component_module(cls, subcls)

    @staticmethod
    def load_trec_run(fn):
        run = defaultdict(dict)
        with open(fn, "rt") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    qid, _, docid, rank, score, desc = line.split(" ")
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn):
        count = 0
        with open(outfn, "wt") as outf:
            for qid in sorted(preds):
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    def exists(self):
        return os.path.exists(os.path.join(self.run_path, "done"))

    def _query_index():
        raise NotImplementedError()

    def create(self):
        self.index.create(self.pipeline_config)
        if self.exists():
            return

        # TODO check for and remove incomplete searcher files? then we can use -skipexists
        self._query_index()
        with open(os.path.join(self.run_path, "done"), "wt") as donef:
            print("done", file=donef)

    def search_run_iter(self, load_run=True):
        self.create()
        for fn in os.listdir(self.run_path):
            if fn == "done" or fn.endswith(".metrics"):
                continue
            fn = os.path.join(self.run_path, fn)
            if load_run:
                yield self.load_trec_run(fn)
            else:
                yield fn

    def search_run_metrics(self, fn, evaluator, qids):
        cachefn = fn + ".metrics"
        try:
            with open(cachefn, "rt") as cachef:
                metrics_cache = json.load(cachef)
        except:
            metrics_cache = {}

        # return from cache if available
        if all(qid in metrics_cache for qid in qids):
            # pytrec_eval does not include qids with no qrels in its output dict, so we also skip them here
            return {qid: metrics_cache[qid] for qid in qids if metrics_cache[qid] != "NONE"}

        # else calculate and cache the metrics
        run = self.load_trec_run(fn)
        metrics = evaluator.evaluate(run)

        # update works for adding missing qids, but will not work for missing metrics
        # (i.e., the old metrics will be overwritten due to dict nesting)
        metrics_cache.update(metrics)
        # add "NONE" placeholders for qids that pytrec_eval omitted (due to having no relevant docs in qrels)
        metrics_cache.update({qid: "NONE" for qid in qids if qid not in metrics_cache})
        with open(cachefn, "wt") as outf:
            json.dump(metrics_cache, outf, indent=4)

        # pytrec_eval does not include qids with no qrels in its output dict, so we also skip them here
        return {qid: metrics_cache[qid] for qid in qids if metrics_cache[qid] != "NONE"}

    def crossvalidated_ranking(self, dev_qids, test_qids, metric="map", full_run=False):
        """ Return a ranking for queries in test_qids using parameters chosen using the queries in dev_qids """

        valid_metrics = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank"}
        cut_points = [5, 10, 15, 20, 30, 100, 200, 500, 1000]
        # the metrics we expect pytrec_eval to output (after expanding _cut)
        expected_metrics = {m for m in valid_metrics if not m.endswith("_cut") and m != "P"} | {
            m + "_" + str(cutoff) for cutoff in cut_points for m in valid_metrics if m.endswith("_cut") or m == "P"
        }

        if metric in ["ndcg", "ndcg_cut"]:
            mkey = "ndcg_cut_20"
        elif metric in expected_metrics:
            mkey = metric
        else:
            raise RuntimeError("requested metric %s is not one of the supported metrics: %s" % (metric, sorted(expected_metrics)))
        avg_metric = lambda run_metrics: np.mean([qid[mkey] for qid in run_metrics.values()])

        dev_qrels = {qid: labels for qid, labels in self.collection.qrels.items() if qid in dev_qids}
        dev_eval = pytrec_eval.RelevanceEvaluator(dev_qrels, valid_metrics)
        best_metric, best_run_fn = -np.inf, None
        for search_run_fn in self.search_run_iter(load_run=False):
            run_metrics = self.search_run_metrics(search_run_fn, dev_eval, dev_qids)
            mavgp = avg_metric(run_metrics)
            # assert that all qids are in what we get back from the evaluator?
            # -> looks like it returns a metric of 0 for qids with all 0 labels, so this should work fine
            if mavgp > best_metric:
                best_metric = mavgp
                best_run_fn = search_run_fn

        best_run = self.load_trec_run(best_run_fn)
        if full_run:
            test_ranking = best_run
        else:
            test_ranking = {qid: v for qid, v in best_run.items() if qid in test_qids}
        return test_ranking

    @staticmethod
    def interpolate_runs(run1, run2, qids, alpha):
        out = {}
        for qid in qids:
            out[qid] = {}
            assert len(run1[qid]) == len(run2[qid])

            min1, max1 = min(run1[qid].values()), max(run1[qid].values())
            min2, max2 = min(run2[qid].values()), max(run2[qid].values())
            mu1, std1 = np.mean([*run1[qid].values()]), np.std([*run1[qid].values()])
            mu2, std2 = np.mean([*run2[qid].values()]), np.std([*run2[qid].values()])
            for docid, score1 in run1[qid].items():
                if docid not in run2[qid]:
                    score2 = min2
                    print(f"WARNING: missing {qid} {docid}")
                else:
                    score2 = run2[qid][docid]
                score1 = (score1 - min1) / (max1 - min1)
                score2 = (score2 - min2) / (max2 - min2)
                # score1 = (score1 - mu1) / std1
                # score2 = (score2 - mu2) / std2
                out[qid][docid] = alpha * score1 + (1 - alpha) * score2

        return out

    @staticmethod
    def crossvalidated_interpolation(dev, test, metric):
        """ Return an interpolated ranking """

        # TODO refactor out (shared with crossvalidated_ranking)
        valid_metrics = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank"}
        cut_points = [5, 10, 15, 20, 30, 100, 200, 500, 1000]
        # the metrics we expect pytrec_eval to output (after expanding _cut)
        expected_metrics = {m for m in valid_metrics if not m.endswith("_cut") and m != "P"} | {
            m + "_" + str(cutoff) for cutoff in cut_points for m in valid_metrics if m.endswith("_cut") or m == "P"
        }

        if metric in ["ndcg", "ndcg_cut"]:
            mkey = "ndcg_cut_20"
        elif metric in expected_metrics:
            mkey = metric
        else:
            raise RuntimeError("requested metric %s is not one of the supported metrics: %s" % (metric, sorted(expected_metrics)))
        avg_metric = lambda run_metrics: np.mean([qid[mkey] for qid in run_metrics.values()])

        assert len(set(dev["qrels"].keys()).intersection(test["qrels"].keys())) == 0
        dev_eval = pytrec_eval.RelevanceEvaluator(dev["qrels"], valid_metrics)
        best_metric, best_alpha = -np.inf, None
        for alpha in np.arange(0, 1.001, 0.05):
            run_metrics = dev_eval.evaluate(
                Searcher.interpolate_runs(dev["reranker"], dev["searcher"], dev["qrels"].keys(), alpha)
            )
            mavgp = avg_metric(run_metrics)
            if mavgp > best_metric:
                best_metric = mavgp
                best_alpha = alpha

        test_run = Searcher.interpolate_runs(test["reranker"], test["searcher"], test["qrels"].keys(), best_alpha)
        dev_run = Searcher.interpolate_runs(dev["reranker"], dev["searcher"], dev["qrels"].keys(), best_alpha)
        return (best_alpha, test_run, dev_run)


import_component_modules("searcher")
