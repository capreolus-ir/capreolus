import os

import torch

from capreolus.sampler import TrainDataset, PredDataset
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH, CACHE_BASE_PATH
from capreolus import evaluator


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    metric = "map"
    fold = config["fold"]

    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]
    searcher["index"].create_index()

    topics_fn = benchmark.topic_file
    searcher_cache_dir = os.path.join(searcher.get_cache_path(), benchmark.name)
    searcher_run_dir = searcher.query_from_file(topics_fn, searcher_cache_dir)

    best_search_run_path = evaluator.search_best_run(searcher_run_dir, benchmark, metric)["path"][fold]
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
    trained_model = reranker["trainer"].train(
        reranker, train_dataset, train_output_path, dev_dataset, dev_output_path, benchmark.qrels
    )
    trained_model.load_best_model(reranker, metric="map")

    test_run = {qid: docs for qid, docs in best_search_run.items() if qid in benchmark.folds[fold]["predict"]["test"]}
    test_dataset = PredDataset(qid_docid_to_rank=test_run, extractor=reranker["extractor"])
    test_output_path = train_output_path / "pred" / "test"

    reranker_pred_fn = reranker["trainer"].predict(trained_model, test_dataset, test_output_path)

    metrics = evaluator.eval_runs(preds, benchmark.qrels, ["ndcg_cut_20", "map", "P_20"])
    return evaluator.evaluate(reranker_pred_fn)


def evaluate(config, modules):
    output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]
    reranker.build()

    metric = "map"
    searcher_output_dir = searcher.get_cache_path() / benchmark.name
    best_search_run_path = evaluator.search_best_run(searcher_output_dir, benchmark, metric)["path"]
    best_search_run = searcher.load_trec_run(best_search_run_path)
    pred_dataset = PredDataset(best_search_run, benchmark, extractor)
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=config["batch"])
    # The reranker's test() method is called here


def _pipeline_path(config, modules):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid", "fold"]}
    pipeline_path = "_".join(["task-rerank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])

    output_path = (
        RESULTS_BASE_PATH
        / config["expid"]
        / modules["collection"].get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / modules["reranker"].get_module_path(include_provided=False)
        / modules["benchmark"].get_module_path()
        / pipeline_path
        / config["fold"]
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
