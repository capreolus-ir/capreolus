import os

import pytrec_eval
import numpy as np

from capreolus.utils.loginit import get_logger
from capreolus.searcher import Searcher

logger = get_logger(__name__)

VALID_METRICS = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank"}
CUT_POINTS = [5, 10, 15, 20, 30, 100, 200, 500, 1000]


def _verify_metric(metric):
    expected_metrics = {m for m in VALID_METRICS if not m.endswith("_cut") and m != "P"} | {
        m + "_" + str(cutoff) for cutoff in CUT_POINTS for m in VALID_METRICS if m.endswith("_cut") or m == "P"
    }
    if metric not in expected_metrics:
        raise ValueError(f"Unexpected evaluation metric: {metric}, should be one of { ' '.join(expected_metrics)}")


def _transform_metric(metric):
    """ Remove the _NUM at the end of metric is applicable """
    if "_cut" in metric or "P_" in metric:
        metric = "_".join(metric.split("_")[:-1])
    return {metric}


def _eval_runfile(runfile, dev_qids, qrels, metric):
    dev_qrels = {qid: labels for qid, labels in qrels.items() if qid in dev_qids}
    evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, _transform_metric(metric))
    run = Searcher.load_trec_run(runfile)
    score = np.mean([metrics_dict.get(metric, -1) for metrics_dict in evaluator.evaluate(run).values()])
    return score


def eval_runfile(runfile, qrels, metric):
    """
    Evaluate single runfile produced by ranker or reranker

    Args:
        runfile: str, path to runfile
        qrels: dict, containing the judgements provided by benchmark
        metric: metric expected to calculate, e.g. ndcg_cut_20, etc

    Returns:
        a dict storing specified metric score and path to the corresponding runfile
    """
    _verify_metric(metric)
    return {
        metric: _eval_runfile(runfile, dev_qids=list(qrels.keys()), qrels=qrels, metric=metric),
        "path": runfile,
    }


def search_best_run(runfile_dir, benchmark, metric, folds=None):
    """
    Select the runfile with respect to the specified metric

    Args:
        runfile_dir: the directory path to all the runfiles to select from
        benchmark: Benchmark class
        metric: str, metric expected to calculate, e.g. ndcg_cut_20, etc
        folds: str, the name of fold to select from

    Returns:
       a dict storing specified metric score and path to the corresponding runfile
    """
    _verify_metric(metric)
    runfiles = [os.path.join(runfile_dir, f) for f in os.listdir(runfile_dir) if f != "done"]
    if len(runfiles) == 1:
        return eval_runfile(runfiles[0], benchmark.qrels, metric)

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    best_scores = {s: {metric: 0, "path": None} for s in [*folds, "avg"]}
    for runfile in runfiles:
        scores = []
        for s, v in folds.items():
            score = _eval_runfile(
                runfile, dev_qids=(set(v["train_qids"]) | set(v["predict"]["dev"])), qrels=benchmark.qrels, metric=metric
            )
            scores.append(score)
            if score > best_scores[s][metric]:
                best_scores[s] = {metric: score, "path": runfile}

        if np.mean(scores) > best_scores["avg"][metric]:
            best_scores["avg"] = {metric: np.mean(scores), "path": runfile}

    key = list(folds.keys())[0] if len(folds) == 1 else "avg"
    return best_scores[key]
