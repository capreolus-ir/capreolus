import os
from collections.abc import Iterable

import numpy as np
import pytrec_eval

from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger
from capreolus.eval.msmarco_eval import compute_metrics_from_files

from ir_measures import *
from ir_measures.measures import Measure

logger = get_logger(__name__)


def format_metrics_string(metrics_scores_dict):
    return " ".join(
        [f"{metric}={v:0.3f}" for metric, v in sorted(metrics_scores_dict.items(), key=lambda kv: (str(kv[0]), kv[1]))]
    )


def log_metrics_verbose(metrics_scores_dict):
    for metric, score in sorted(metrics_scores_dict.items(), key=lambda kv: (str(kv[0]), kv[1])):
        logger.info("%25s: %0.4f", metric, score)


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
    assert isinstance(
        primary_metric, Measure
    ), f"Expect primary metric to be ir_measures.measures.Meature, but got str: {primary_metric}"
    assert (
        (metrics is None)
        or isinstance(metrics, Measure)
        or (isinstance(metrics, Iterable) and all(isinstance(m, Measure) for m in metrics))
    ), f"Expect metrics to be either ir_measures.measures.Meature or a series of ir_measures.measures.Meature, but got str: {metrics}"

    metrics = [] if not metrics else ([metrics] if isinstance(metrics, Measure) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics
    assert all(isinstance(m, Measure) for m in metrics)

    test_runs = {}
    alphas = {}
    for s, v in benchmark.folds.items():
        best_metric = None
        dev_qids = set(v["predict"]["dev"])
        dev1, dev2 = run1[s]["dev"], run2[s]["dev"]

        for alpha in np.arange(0, 1.001, 0.05):
            interpolated_run = interpolate_runs(dev1, dev2, dev_qids, alpha)
            scores = benchmark.evaluate(interpolated_run, metrics=metrics)

            if best_metric is None or scores[primary_metric] > best_metric:
                best_metric = scores[primary_metric]
                alphas[s] = alpha

        test_qids = set(v["predict"]["test"])
        test1, test2 = run1[s]["test"], run2[s]["test"]
        interpolated_test_run = interpolate_runs(test1, test2, test_qids, alphas[s])
        for qid in test_qids:
            assert qid not in test_runs
            test_runs[qid] = interpolated_test_run[qid].copy()

    scores = benchmark.evaluate(test_runs, metrics=metrics)

    return {"score": scores, "alphas": alphas}
