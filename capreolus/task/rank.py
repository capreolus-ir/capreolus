from profane import ModuleBase, Dependency, ConfigOption, constants

from capreolus import evaluator
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class RankTask(Task):
    module_name = "rank"
    requires_random_seed = False
    config_spec = [ConfigOption("optimize", "map", "metric to maximize on the dev set"), ConfigOption("filter", False)]
    config_keys_not_in_path = ["optimize"]  # only used for choosing best result; does not affect search()
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="wsdm20demo", provide_this=True, provide_children=["collection"]),
        Dependency(key="searcher", module="searcher", name="BM25"),
    ]

    commands = ["run", "evaluate"] + Task.help_commands
    default_command = "describe"

    def search(self):
        topics_fn = self.benchmark.topic_file
        output_dir = self.get_results_path()

        if hasattr(self.searcher, "index"):
            self.searcher.index.create_index()

        if self.config["filter"]:
            qrels = load_qrels(self.benchmark.qrel_ignore)
            docs_to_remove = {q: list(d.keys()) for q, d in qrels.items()}
            search_results_folder = self.searcher.query_from_file(topics_fn, output_dir, docs_to_remove)
        else:
            search_results_folder = self.searcher.query_from_file(topics_fn, output_dir)

        logger.info("searcher results written to: %s", search_results_folder)
        return search_results_folder

    def evaluate(self):
        best_results = evaluator.search_best_run(
            self.get_results_path(), self.benchmark, primary_metric=self.config["optimize"], metrics=evaluator.DEFAULT_METRICS
        )

        for fold, path in best_results["path"].items():
            logger.info("rank: fold=%s best run: %s", fold, path)

        logger.info("rank: cross-validated results when optimizing for '%s':", self.config["optimize"])
        for metric, score in sorted(best_results["score"].items()):
            logger.info("%15s: %0.4f", metric, score)

        return best_results
