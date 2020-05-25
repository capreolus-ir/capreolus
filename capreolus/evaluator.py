import os

import pytrec_eval
import numpy as np

from capreolus.utils.loginit import get_logger
from capreolus.searcher import Searcher

logger = get_logger(__name__)

VALID_METRICS = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank", "set_recall"}
CUT_POINTS = [5, 10, 15, 20, 30, 100, 200, 500, 1000]


def mrr(qrels, runs, qids=None):
    if qids:
        qrels = {q: v for q, v in qrels.items() if q in qids}
        runs = {q: v for q, v in runs.items() if q in qids}

    ranks = []
    for q, rundocs in runs.items():
        if q not in qrels:
            continue

        rundocs = sorted(rundocs.items(), key=lambda k_v: float(k_v[1]), reverse=True)
        rundocs = [d for d, i in rundocs]
        pos_docids, pos_doc_ranks = [d for d in rundocs if qrels[q].get(d, 0) > 0], []
        for d in pos_docids:
            if d in rundocs:
                pos_doc_ranks.append(rundocs.index(d) + 1)
        ranks.append(1 / min(pos_doc_ranks)) if len(pos_doc_ranks) > 0 else 0
    return sum(ranks) / len(ranks)


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
            raise ValueError(f"Unexpected evaluation metric: {metric}, should be one of { ' '.join(sorted(expected_metrics))}")


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
    calc_mrr = "mrr" in metrics
    if calc_mrr:
        metrics.remove("mrr")
        mrr_score = mrr(qrels, runs, dev_qids)

    _verify_metric(metrics)

    dev_qrels = {qid: labels for qid, labels in qrels.items() if qid in dev_qids}
    evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, _transform_metric(metrics))

    scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in evaluator.evaluate(runs).values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(metrics, scores))

    if calc_mrr:
        scores["mrr"] = mrr_score

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
    runs = Searcher.load_trec_run(runfile)
    return _eval_runs(runs, qrels, metrics, dev_qids=list(qrels.keys()))


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
        return {"score": eval_runfile(runfiles[0], benchmark.qrels, metrics), "path": {s: runfiles[0] for s in folds}}

    best_scores = {s: {primary_metric: 0, "path": None} for s in folds}
    for runfile in runfiles:
        runs = Searcher.load_trec_run(runfile)
        for s, v in folds.items():
            score = _eval_runs(
                runs, benchmark.qrels, [primary_metric], dev_qids=(set(v["train_qids"]) | set(v["predict"]["dev"]))
            )[primary_metric]
            if score > best_scores[s][primary_metric]:
                best_scores[s] = {primary_metric: score, "path": runfile}

    test_runs, test_qrels = {}, {}
    for s, score_dict in best_scores.items():
        test_qids = folds[s]["predict"]["test"]
        test_runs.update({qid: v for qid, v in Searcher.load_trec_run(score_dict["path"]).items() if qid in test_qids})
        test_qrels.update({qid: v for qid, v in benchmark.qrels.items() if qid in test_qids})

    scores = eval_runs(test_runs, benchmark.qrels, metrics)
    return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}
