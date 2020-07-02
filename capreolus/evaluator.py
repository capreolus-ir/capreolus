import os

import numpy as np
import pytrec_eval

from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = [
    "P_1",
    "P_5",
    "P_10",
    "P_20",
    "judged_10",
    "judged_20",
    "judged_200",
    "map",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "recall_100",
    "recall_1000",
    "recip_rank",
]


def judged(qrels, runs, n):
    scores = []
    for q, rundocs in runs.items():
        if q not in qrels:
            logger.error(f"{q} in run files cannot be found in qrels")
            continue

        if len(rundocs) == 0:
            scores.append(0)
            continue

        topn = sorted(rundocs.keys(), key=rundocs.get, reverse=True)[:n]
        score = sum(docid in qrels[q] for docid in topn) / len(topn)
        scores.append(score)

    return sum(scores) / len(scores)


def _eval_runs(runs, qrels, metrics, dev_qids, relevance_level):
    assert isinstance(metrics, list)
    calc_judged = [int(metric.split("_")[1]) for metric in metrics if metric.startswith("judged_")]
    for n in calc_judged:
        metrics.remove(f"judged_{n}")

    dev_qrels = {qid: labels for qid, labels in qrels.items() if qid in dev_qids}
    evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, metrics, relevance_level=int(relevance_level))
    scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in evaluator.evaluate(runs).values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(metrics, scores))

    for n in calc_judged:
        scores[f"judged_{n}"] = judged(qrels, runs, n)

    return scores


def eval_runs(runs, qrels, metrics, relevance_level=1):
    """
    Evaluate runs produced by a ranker (or loaded with Searcher.load_trec_run)

    Args:
        runs: dict in the format ``{qid: {docid: score}}``
        qrels: dict containing relevance judgements (e.g., ``benchmark.qrels``)
        metrics (str or list): metrics to calculate (e.g., ``evaluator.DEFAULT_METRICS``)
        relevance_level (int): relevance label threshold to use with non-graded metrics (equivalent to trec_eval's --level_for_rel)

    Returns:
           dict: a dict in the format ``{metric: score}`` containing the average score for each metric
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    return _eval_runs(runs, qrels, metrics, list(qrels.keys()), relevance_level)


def eval_runfile(runfile, qrels, metrics, relevance_level):
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
    runs = Searcher.load_trec_run(runfile)
    return _eval_runs(runs, qrels, metrics, list(qrels.keys()), relevance_level)


def search_best_run(runfile_dirs, benchmark, primary_metric, metrics=None, folds=None):
    """
    Select the runfile with respect to the specified metric

    Args:
        runfile_dirs: the directory path to all the runfiles to select from
        benchmark: Benchmark class
        primary_metric: str, metric used to select the best runfile , e.g. ndcg_cut_20, etc
        metrics: str or list, metric expected by be calculated on the best runs
        folds: str, the name of fold to select from

    Returns:
       a dict storing specified metric score and path to the corresponding runfile
    """

    if not isinstance(runfile_dirs, (list, tuple)):
        runfile_dirs = [runfile_dirs]

    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    runfiles = [
        os.path.join(runfile_dir, f)
        for runfile_dir in runfile_dirs
        for f in os.listdir(runfile_dir)
        if (f != "done" and not os.path.isdir(os.path.join(runfile_dir, f)))
    ]

    best_scores = {s: {primary_metric: 0, "path": None} for s in folds}
    for runfile in runfiles:
        runs = Searcher.load_trec_run(runfile)
        for s, v in folds.items():
            score = _eval_runs(
                runs,
                benchmark.qrels,
                [primary_metric],
                (set(v["train_qids"]) | set(v["predict"]["dev"])),
                benchmark.relevance_level,
            )[primary_metric]
            if score > best_scores[s][primary_metric]:
                best_scores[s] = {primary_metric: score, "path": runfile}

    test_runs = {}
    for s, score_dict in best_scores.items():
        test_qids = folds[s]["predict"]["test"]
        # any empty (no results) queries need to be added so they contribute zeros to the average
        test_runs.update({qid: {} for qid in test_qids})
        test_runs.update({qid: v for qid, v in Searcher.load_trec_run(score_dict["path"]).items() if qid in test_qids})

    scores = eval_runs(test_runs, benchmark.qrels, metrics, benchmark.relevance_level)
    return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}


def interpolate_runs(run1, run2, qids, alpha):
    out = {}
    for qid in qids:
        out[qid] = {}

        if len(run1[qid]) == 0:
            min1, max1 = 0, 1
        else:
            min1, max1 = min(run1[qid].values()), max(run1[qid].values())

            if min1 == max1:
                min1 = 0.01 * max1 - 0.01

        if len(run2[qid]) == 0:
            min2, max2 = 0, 1
        else:
            min2, max2 = min(run2[qid].values()), max(run2[qid].values())

            if min2 == max2:
                min2 = 0.01 * max2 - 0.01

        for docid in run1[qid].keys() | run2[qid]:
            score1 = run1[qid].get(docid, min1)
            score2 = run2[qid].get(docid, min2)

            score1 = (score1 - min1) / (max1 - min1)
            score2 = (score2 - min2) / (max2 - min2)
            out[qid][docid] = alpha * score1 + (1 - alpha) * score2

    return out


def interpolated_eval(run1, run2, benchmark, primary_metric, metrics=None):
    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    test_runs = {}
    alphas = {}
    for s, v in benchmark.folds.items():
        best_metric = None
        dev_qids = set(v["predict"]["dev"])
        dev1, dev2 = run1[s]["dev"], run2[s]["dev"]

        for alpha in np.arange(0, 1.001, 0.05):
            interpolated_run = interpolate_runs(dev1, dev2, dev_qids, alpha)
            metrics = eval_runs(interpolated_run, benchmark.qrels, metrics, benchmark.relevance_level)

            if best_metric is None or metrics[primary_metric] > best_metric:
                best_metric = metrics[primary_metric]
                alphas[s] = alpha

        test_qids = set(v["predict"]["test"])
        test1, test2 = run1[s]["test"], run2[s]["test"]
        interpolated_test_run = interpolate_runs(test1, test2, test_qids, alphas[s])
        for qid in test_qids:
            assert qid not in test_runs
            test_runs[qid] = interpolated_test_run[qid].copy()

    scores = eval_runs(test_runs, benchmark.qrels, metrics, benchmark.relevance_level)
    return {"score": scores, "alphas": alphas}
