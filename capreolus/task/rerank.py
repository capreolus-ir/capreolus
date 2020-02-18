import os
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    output_path = _pipeline_path(config, modules)
    print("**** got train")


def evaluate(config, modules):
    output_path = _pipeline_path(config, modules)
    print("**** got evaluate!!")


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
