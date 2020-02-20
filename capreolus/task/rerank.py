import json
import os

import torch

from capreolus.sampler import TrainDataset, PredDataset
from capreolus.task import Task
from capreolus.registry import print_module_graph, RESULTS_BASE_PATH
from capreolus import evaluator


def describe(config, modules):
    print("\n--- module dependency graph ---")
    print("run.py")
    for module, obj in modules.items():
        print_module_graph(obj, prefix=" ")
    print("-------------------------------")

    print("\n\n--- config: ---")
    print(json.dumps(config, indent=4))

    # prepare an output path that contains all config options
    # experiment_id / collection / benchmark / [[index/searcher]] / [[index/extractor/reranker]] / pytorch-pipeline / <fold>
    output_path = _pipeline_path(config, modules)
    print("\n\nresults path:", output_path)

    print("cache paths:")
    for module, obj in modules.items():
        print("  ", obj.get_cache_path())

    searcher = modules["searcher"]
    reranker = modules["reranker"]
    benchmark = modules["benchmark"]

    # now we can use the modules via their APIs (which still need to be defined)
    # ... set up a training loop ...
    # ... for batch in benchmark.datagen ...
    # ... reranker.forward(batch) ...


def train(config, modules):
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    reranker = modules["reranker"]
    extractor = reranker.modules["extractor"]

    topics_fn = benchmark.topic_file
    metric = "map"
    searcher.index.create_index()
    search_results_folder = searcher.query_from_file(topics_fn, os.path.join(searcher.get_cache_path(), benchmark.name))
    best_search_run_path = evaluator.search_best_run(search_results_folder, benchmark, metric)["path"]
    best_search_run = searcher.load_trec_run(best_search_run_path)
    train_dataset = TrainDataset(best_search_run, benchmark, extractor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch"])
    # Call the reranker's score() method here


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

        # ... pytorch stuff ...
        batch = 32

    name = "rerank"
    module_order = ["collection", "searcher", "reranker", "benchmark"]
    module_defaults = {"searcher": "BM25", "reranker": "KNRM", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
