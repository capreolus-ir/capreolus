import os
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH

from capreolus import evaluator


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    # output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    topics_fn = benchmark.topic_file

    searcher["index"].create_index()

    benchmark_dirname = benchmark.name + "_".join([f"{k}={v}" for k, v in benchmark.cfg.items() if k != "_name"])
    output_dir = searcher.get_cache_path() / benchmark_dirname
    search_results_folder = searcher.query_from_file(topics_fn, output_dir)
    print(f"Search results are at: {search_results_folder}")


def evaluate(config, modules):
    # output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    benchmark_dirname = benchmark.name + "_".join([f"{k}={v}" for k, v in benchmark.cfg.items() if k != "_name"])

    metric = config["optimize"]
    all_metric = ["mrr", "ndcg_cut_20", "ndcg_cut_10", "map", "P_20", "P_10", "set_recall"]
    output_dir = searcher.get_cache_path() / benchmark_dirname
    best_results = evaluator.search_best_run(output_dir, benchmark, primary_metric=metric, metrics=all_metric)
    pathes = [f"\t{s}: {path}" for s, path in best_results["path"].items()]
    print("path for each split: \n", "\n".join(pathes))

    scores = [f"\t{s}: {score}" for s, score in best_results["score"].items()]
    print(f"cross-validated results when optimizing for {metric}: \n", "\n".join(scores))


def _pipeline_path(config, modules):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid"]}
    pipeline_path = "_".join(["task-rank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])
    output_path = (
        RESULTS_BASE_PATH
        / config["expid"]
        / modules["collection"].get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / pipeline_path
        / modules["benchmark"].get_module_path()
    )

    return output_path


class RankTask(Task):
    def pipeline_config():
        expid = "debug"
        seed = 123_456
        # eval_metrics = {"map", "ndcg_cut_20", "ndcg_cut_10", "P_20"}
        optimize = "map"  # metric to maximize on the dev set

    name = "rank"
    module_order = ["collection", "searcher", "benchmark"]
    module_defaults = {"searcher": "BM25", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
