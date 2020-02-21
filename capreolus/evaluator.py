import os

import pytrec_eval
import numpy as np

from capreolus.utils.loginit import get_logger
from capreolus.searcher import Searcher

logger = get_logger(__name__)

VALID_METRICS = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank"}
CUT_POINTS = [5, 10, 15, 20, 30, 100, 200, 500, 1000]


def _verify_metric(metrics):
    """
    Verify if the metrics is in the returned list of TREC eval

    Args:
        metrics: a list of str
    """
    assert isinstance(metrics, list)
    expected_metrics = {m for m in VALID_METRICS if not m.endswith("_cut") and m != "P"} | {
        m + "_" + str(cutoff) for cutoff in CUT_POINTS for m in VALID_METRICS if m.endswith("_cut") or m == "P"
    }
    for metric in metrics:
        if metric not in expected_metrics:
            raise ValueError(f"Unexpected evaluation metric: {metric}, should be one of { ' '.join(expected_metrics)}")


def _transform_metric(metrics):
    """
    Remove the _NUM at the end of metric is applicable

    Args:
        metrics: a list of str

    Returns:
        a set of transformed metric
    """
    assert isinstance(metrics, list)
    metrics = {"_".join(metric.split("_")[:-1]) if "_cut" in metric or "P_" in metric else metric for metric in metrics}
    return metrics


def _eval_runs(runs, qrels, metrics, dev_qids):
    assert isinstance(metrics, list)
    dev_qrels = {qid: labels for qid, labels in qrels.items() if qid in dev_qids}
    evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, _transform_metric(metrics))

    tmpx = evaluator.evaluate(runs).values()
    scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in tmpx]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(metrics, scores))
    return scores


def eval_runs(runs, qrels, metrics):
    """
    Evaluate runs loaded by Searcher.load_trec_run

    Args:
        runs: a dict with format {qid: {docid: score}}, could be prepared by Searcher.load_trec_run
        qrels: dict, containing the judgements provided by benchmark
        metrics: str or list, metrics expected to calculate, e.g. ndcg_cut_20, etc

    Returns:
        a dict with format {metric: score}, containing the evaluation score of specified metrics
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    _verify_metric(metrics)
    return _eval_runs(runs, qrels, metrics, dev_qids=list(qrels.keys()))


def eval_runfile(runfile, qrels, metrics):
    """
    Evaluate a single runfile produced by ranker or reranker

    Args:
        runfile: str, path to runfile
        qrels: dict, containing the judgements provided by benchmark
        metrics: str or list, metrics expected to calculate, e.g. ndcg_cut_20, etc

    Returns:
        a dict with format {metric: score}, containing the evaluation score of specified metrics
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    _verify_metric(metrics)
    runs = Searcher.load_trec_run(runfile)
    # return {metric: _eval_runfile(runfile, dev_qids=list(qrels.keys()), qrels=qrels, metric=metric), "path": runfile}
    return _eval_runs(runs, qrels, metrics, dev_qids=list(qrels.keys()))


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
    _verify_metric([metric])
    runfiles = [os.path.join(runfile_dir, f) for f in os.listdir(runfile_dir) if f != "done"]
    if len(runfiles) == 1:
        return {metric: eval_runfile(runfiles[0], benchmark.qrels, metric)[metric], "path": runfiles[0]}

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    best_scores = {s: {metric: 0, "path": None} for s in [*folds, "avg"]}
    for runfile in runfiles:
        scores = []
        runs = Searcher.load_trec_run(runfile)
        for s, v in folds.items():
            score = _eval_runs(
                runs, benchmark.qrels, [metric],
                dev_qids=(set(v["train_qids"]) | set(v["predict"]["dev"])),
            )[metric]
            scores.append(score)
            if score > best_scores[s][metric]:
                best_scores[s] = {metric: score, "path": runfile}

        if np.mean(scores) > best_scores["avg"][metric]:
            best_scores["avg"] = {metric: np.mean(scores), "path": runfile}

    key = list(folds.keys())[0] if len(folds) == 1 else "avg"
    best_scores = best_scores[key]
    best_scores[metric] = eval_runfile(best_scores["path"], benchmark.qrels, metric)[metric]
    return best_scores
