import os

import torch

from capreolus.sampler import TrainDataset, PredDataset
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH
from capreolus import evaluator


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    output_path = _pipeline_path(config, modules)
    metric = "map"

    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]
    evaluator = modules["evaluate"]
    searcher["index"].create_index()

    topics_fn = benchmark.topic_file
    searcher_cache_dir = os.path.join(searcher.get_cache_path(), benchmark.name)
    searcher_run_dir = searcher.query_from_file(topics_fn, searcher_cache_dir)

    best_search_run_path = evaluator.search_best_run(searcher_run_dir, benchmark, metric)["path"]
    best_search_run = searcher.load_trec_run(best_search_run_path)

    # create train/pred pairs here (not in benchmark)
    # train_pairs, pred_pairs = benchmark.create_train_pred_pairs(best_search_run_fn)

    # (query_text, posdoc, negdoc) - samplerreturns the actual query text _and_ the qid

    # TODO API in flux
    train_sampler = sampler.get_train_sampler(best_search_run_fn, benchmark, reranker.extractor)
    dev_instances = sampler.get_dev_iterator(best_search_run_fn, benchmark, reranker.extractor)
    test_instances = sampler.get_test_iterator(best_search_run_fn, benchmark, reranker.extractor)

    train_dataset = TrainDataset(best_search_run, benchmark, extractor)
    trained_model = trainer.train(reranker, train_sampler, dev_instances, weights_path)

    trained_model.load_best_model(reranker, metric="map")
    reranker_pred_fn = trainer.predict(trained_model, test_instances, output_fn)

    return evaluator.evaluate(reranker_pred_fn)


def evaluate(config, modules):
    output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]
    extractor = reranker.modules["extractor"]

    metric = "map"
    searcher_output_dir = searcher.get_cache_path() / benchmark.name
    best_search_run_path = evaluator.search_best_run(searcher_output_dir, benchmark, metric)["path"]
    best_search_run = searcher.load_trec_run(best_search_run_path)
    pred_dataset = PredDataset(best_search_run, benchmark, extractor)
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=config["batch"])
    # The reranker's test() method is called here


def _pipeline_path(config, modules):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid"]}
    pipeline_path = "_".join(["task-rerank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])
    output_path = (
        RESULTS_BASE_PATH
        / config["expid"]
        / modules["collection"].get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / modules["reranker"].get_module_path(include_provided=False)
        / pipeline_path
        / modules["benchmark"].get_module_path()
    )
    return output_path


class RerankTask(Task):
    def pipeline_config():
        expid = "debug"
        seed = 123_456

    name = "rerank"
    module_order = ["collection", "searcher", "reranker", "benchmark"]
    module_defaults = {"searcher": "BM25", "reranker": "KNRM", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
