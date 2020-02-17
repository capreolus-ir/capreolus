import json
import os
from capreolus.task import Task
from capreolus.registry import print_module_graph, RESULTS_BASE_PATH


def describe(config, modules):
    print("\n--- module dependency graph ---")
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
    benchmark = modules["benchmark"]


def train(config, modules):
    output_path = _pipeline_path(config, modules)
    print("**** got train")
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]

    searcher["index"].create_index()
    print(searcher["index"].getdoc("FBIS4-16592"))


def evaluate(config, modules):
    output_path = _pipeline_path(config, modules)
    print("**** got evaluate!!")


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

    name = "rank"
    module_order = ["collection", "searcher", "benchmark"]
    module_defaults = {"searcher": "BM25", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
