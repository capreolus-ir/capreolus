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
    evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, metrics, relevance_level=relevance_level)

    scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in evaluator.evaluate(runs).values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(metrics, scores))

    for n in calc_judged:
        scores[f"judged_{n}"] = judged(qrels, runs, n)

    return scores


def eval_runs(runs, qrels, metrics, relevance_level):
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


def search_best_run(runfile_dir, benchmark, primary_metric, metrics=None, folds=None):
    """
    Select the runfile with respect to the specified metric

    Args:
        runfile_dir: the directory path to all the runfiles to select from
        benchmark: Benchmark class
        primary_metric: str, metric used to select the best runfile , e.g. ndcg_cut_20, etc
        metrics: str or list, metric expected by be calculated on the best runs
        folds: str, the name of fold to select from

    Returns:
       a dict storing specified metric score and path to the corresponding runfile
    """
    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    runfiles = [
        os.path.join(runfile_dir, f)
        for f in os.listdir(runfile_dir)
        if (f != "done" and not os.path.isdir(os.path.join(runfile_dir, f)))
    ]

    if len(runfiles) == 1:
        return {
            "score": eval_runfile(runfiles[0], benchmark.qrels, metrics, benchmark.relevance_level),
            "path": {s: runfiles[0] for s in folds},
        }

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

    test_runs, test_qrels = {}, {}
    for s, score_dict in best_scores.items():
        test_qids = folds[s]["predict"]["test"]
        test_runs.update({qid: v for qid, v in Searcher.load_trec_run(score_dict["path"]).items() if qid in test_qids})
        test_qrels.update({qid: v for qid, v in benchmark.qrels.items() if qid in test_qids})

    scores = eval_runs(test_runs, benchmark.qrels, metrics, benchmark.relevance_level)
    return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}
