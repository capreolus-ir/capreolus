import random
import os


import numpy as np
import torch

from capreolus.sampler import TrainDataset, PredDataset
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH, CACHE_BASE_PATH
from capreolus import evaluator
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    metric = "map"
    fold = config["fold"]

    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]

    if "index" in searcher.modules:
        searcher["index"].create_index()

    topics_fn = benchmark.topic_file
    searcher_cache_dir = os.path.join(searcher.get_cache_path(), benchmark.name)
    searcher_run_dir = searcher.query_from_file(topics_fn, searcher_cache_dir)

    results = evaluator.search_best_run(searcher_run_dir, benchmark, metric)
    print("score: ", results["score"])
    # end of tmp
    best_search_run_path = results["path"][fold]
    best_search_run = searcher.load_trec_run(best_search_run_path)

    docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
    reranker["extractor"].create(qids=best_search_run.keys(), docids=docids, topics=benchmark.topics[benchmark.query_type])
    reranker.build()

    train_run = {qid: docs for qid, docs in best_search_run.items() if qid in benchmark.folds[fold]["train_qids"]}
    dev_run = {qid: docs for qid, docs in best_search_run.items() if qid in benchmark.folds[fold]["predict"]["dev"]}

    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=reranker["extractor"])
    dev_dataset = PredDataset(qid_docid_to_rank=dev_run, extractor=reranker["extractor"])

    train_output_path = _pipeline_path(config, modules)
    dev_output_path = train_output_path / "pred" / "dev"
    reranker["trainer"].train(reranker, train_dataset, train_output_path, dev_dataset, dev_output_path, benchmark.qrels, metric)


def evaluate(config, modules):
    metric = "map"
    fold = config["fold"]
    train_output_path = _pipeline_path(config, modules)
    test_output_path = train_output_path / "pred" / "test" / "best"

    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]

    if os.path.exists(test_output_path):
        test_preds = Searcher.load_trec_run(test_output_path)
    else:
        topics_fn = benchmark.topic_file
        searcher_cache_dir = os.path.join(searcher.get_cache_path(), benchmark.name)
        searcher_run_dir = searcher.query_from_file(topics_fn, searcher_cache_dir)

        best_search_run_path = evaluator.search_best_run(searcher_run_dir, benchmark, metric)["path"][fold]
        best_search_run = searcher.load_trec_run(best_search_run_path)

        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
        reranker["extractor"].create(qids=best_search_run.keys(), docids=docids, topics=benchmark.topics[benchmark.query_type])
        reranker.build()

        reranker["trainer"].load_best_model(reranker, train_output_path)

        test_run = {qid: docs for qid, docs in best_search_run.items() if qid in benchmark.folds[fold]["predict"]["test"]}
        test_dataset = PredDataset(qid_docid_to_rank=test_run, extractor=reranker["extractor"])

        test_preds = reranker["trainer"].predict(reranker, test_dataset, test_output_path)

    metrics = evaluator.eval_runs(test_preds, benchmark.qrels, ["ndcg_cut_20", "ndcg_cut_10", "map", "P_20", "P_10"])
    print("test metrics for fold=%s:" % fold, metrics)

    print("\ncomputing metrics across all folds")
    avg = {}
    found = 0
    for fold in benchmark.folds:
        pred_path = _pipeline_path(config, modules, fold=fold) / "pred" / "test" / "best"
        if not os.path.exists(pred_path):
            print("\tfold=%s results are missing and will not be included" % fold)
            continue

        found += 1
        preds = Searcher.load_trec_run(pred_path)
        metrics = evaluator.eval_runs(preds, benchmark.qrels, ["ndcg_cut_20", "ndcg_cut_10", "map", "P_20", "P_10"])
        for metric, val in metrics.items():
            avg.setdefault(metric, []).append(val)

    avg = {k: np.mean(v) for k, v in avg.items()}
    print(f"average metrics across {found}/{len(benchmark.folds)} folds:", avg)


def _pipeline_path(config, modules, fold=None):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid", "fold"]}
    pipeline_path = "_".join(["task-rerank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])

    if not fold:
        fold = config["fold"]

    output_path = (
        RESULTS_BASE_PATH
        / config["expid"]
        / modules["collection"].get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / modules["reranker"].get_module_path(include_provided=False)
        / modules["benchmark"].get_module_path()
        / pipeline_path
        / fold
    )
    return output_path


class RerankTask(Task):
    def pipeline_config():
        expid = "debug"
        seed = 123_456
        fold = "s1"
        # rundocsonly = True  # use only docs from the searcher as pos/neg training instances (i.e., not all qrels)

    name = "rerank"
    module_order = ["collection", "searcher", "reranker", "benchmark"]
    module_defaults = {"searcher": "BM25", "reranker": "KNRM", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
