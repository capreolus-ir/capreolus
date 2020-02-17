import os

import pytrec_eval
import numpy as np

from capreolus.searcher import Searcher

def score(runfile, benchmark, metrics):
    """
    Args:
        runfile: a str, path to a runfile either from ranker or reranker
        qrels: a dict with the format of {qid: {docid: label}}
        target_metrics: a str or a list of str, specifying all the expect evaluation metrics

    Returns:
        a dict of calculated metric
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    evaluator = pytrec_eval.RelevanceEvaluator(benchmark.qrels, {"map", "ndcg_cut", "P"})
    run = Searcher.load_trec_run(runfile)
    scores = [{m: metrics_dict[m] for m in metrics} for metrics_dict in evaluator.evaluate(run).values()]
    avg_score = {m: np.mean([s[m] for s in scores]) for m in metrics}
    return avg_score

# def search_best_run(runfiles, qrels, metric="ndcg20"):
#     best = { "path": "", "params": "" }
#     best[metric] = 0
#     for runfile in os.listdir(runfiles):
#         scores = evaluate(runfile, qrels)
#
#         if scores[metric] > best[metric]:
#             best["path"] = runfile
#             best["params"] = xxx
#             best[metric] = scores[metric]
