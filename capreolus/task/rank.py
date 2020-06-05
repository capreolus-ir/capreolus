import os

from profane import ModuleBase, Dependency, ConfigOption, constants

from capreolus.task import Task
from capreolus.utils.trec import load_qrels
from capreolus import evaluator


def train(config, modules, output_dir):
    # output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    topics_fn = benchmark.topic_file

    searcher.index.create_index()

    # benchmark_dirname = benchmark.name + "_".join([f"{k}={v}" for k, v in benchmark.config.items() if k != "name"])
    # output_dir = searcher.get_cache_path() / benchmark_dirname

    if config["filter"]:
        qrels = load_qrels(benchmark.qrel_ignore)
        docs_to_remove = {q: list(d.keys()) for q, d in qrels.items()}
        search_results_folder = searcher.query_from_file(topics_fn, output_dir, docs_to_remove)
    else:
        search_results_folder = searcher.query_from_file(topics_fn, output_dir)

    print(f"Search results are at: {search_results_folder}")


def evaluate(config, modules, output_dir):
    # output_path = _pipeline_path(config, modules)
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]
    # benchmark_dirname = benchmark.name + "_".join([f"{k}={v}" for k, v in benchmark.config.items() if k != "name"])

    metric = config["optimize"]
    all_metric = ["mrr", "P_1", "ndcg_cut_20", "ndcg_cut_10", "map", "P_20", "P_10", "set_recall"]
    # output_dir = searcher.get_cache_path() / benchmark_dirname
    best_results = evaluator.search_best_run(output_dir, benchmark, primary_metric=metric, metrics=all_metric)
    pathes = [f"\t{s}: {path}" for s, path in best_results["path"].items()]
    print("path for each split: \n", "\n".join(pathes))

    scores = [f"\t{s}: {score}" for s, score in best_results["score"].items()]
    print(f"cross-validated results when optimizing for {metric}: \n", "\n".join(scores))


def _pipeline_path(config, modules):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid"]}
    pipeline_path = "_".join(["task-rank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])
    output_path = (
        constants["RESULTS_BASE_PATH"]
        / config["expid"]
        / modules["benchmark"].collection.get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / pipeline_path
        / modules["benchmark"].get_module_path()
    )

    return output_path


@Task.register
class RankTask(Task):
    module_name = "rank"
    config_spec = [
        ConfigOption(key="expid", default_value="debug", description="experiment ID"),
        ConfigOption("optimize", "map", "metric to maximzie on the dev set"),
        ConfigOption("filter", False),
    ]
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="wsdm20demo", provide_this=True, provide_children=["collection"]),
        Dependency(key="searcher", module="searcher", name="BM25"),
    ]

    commands = ["train", "evaluate"] + Task.help_commands
    default_command = "describe"

    def train(self):
        # TODO move Task commands inside class
        return train(self.config, self._dependency_objects, self.get_results_path())

    def evaluate(self):
        # TODO move Task commands inside class
        return evaluate(self.config, self._dependency_objects, self.get_results_path())
