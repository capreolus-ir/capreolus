from profane import ConfigOption, Dependency

from capreolus import evaluator
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class RankTask(Task):
    module_name = "rank"
    requires_random_seed = False
    config_spec = [
        ConfigOption("filter", False),
        ConfigOption("optimize", "map", "metric to maximize on the dev set"),
        ConfigOption("metrics", "default", "metrics reported for evaluation", value_type="strlist"),
    ]
    config_keys_not_in_path = ["optimize", "metrics"]  # affect only evaluation but not search()

    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="robust04.yang19", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="searcher", module="searcher", name="BM25"),
    ]

    commands = ["run", "evaluate", "searcheval"] + Task.help_commands
    default_command = "describe"

    def searcheval(self):
        self.search()
        self.evaluate()

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
        metrics = self.config["metrics"] if list(self.config["metrics"]) != ["default"] else evaluator.DEFAULT_METRICS

        best_results = evaluator.search_best_run(
            self.get_results_path(), self.benchmark, primary_metric=self.config["optimize"], metrics=metrics
        )

        for fold, path in best_results["path"].items():
            logger.info("rank: fold=%s best run: %s", fold, path)

        logger.info("rank: cross-validated results when optimizing for '%s':", self.config["optimize"])
        for metric, score in sorted(best_results["score"].items()):
            logger.info("%25s: %0.4f", metric, score)

        return best_results
