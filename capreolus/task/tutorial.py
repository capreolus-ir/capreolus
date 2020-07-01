from capreolus import ConfigOption, Dependency, evaluator
from capreolus.task import Task
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class TutorialTask(Task):
    module_name = "tutorial"
    config_spec = [ConfigOption("optimize", "map", "metric to maximize on the validation set")]
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="nf", provide_this=True, provide_children=["collection"]),
        Dependency(key="searcher1", module="searcher", name="BM25RM3"),
        Dependency(key="searcher2", module="searcher", name="SDM"),
    ]

    commands = ["run"] + Task.help_commands
    default_command = "run"

    def run(self):
        output_dir = self.get_results_path()

        # read the title queries from the chosen benchmark's topic file
        results1 = self.searcher1.query_from_file(self.benchmark.topic_file, output_dir / "searcher1")
        results2 = self.searcher2.query_from_file(self.benchmark.topic_file, output_dir / "searcher2")
        searcher_results = [results1, results2]

        # using the benchmark's folds, which each contain train/validation/test queries,
        # choose the best run in `output_dir` for the fold based on the validation queries
        # and return metrics calculated on the test queries
        best_results = evaluator.search_best_run(
            searcher_results, self.benchmark, primary_metric=self.config["optimize"], metrics=evaluator.DEFAULT_METRICS
        )

        for fold, path in best_results["path"].items():
            shortpath = "..." + path[-40:]
            logger.info("fold=%s best run: %s", fold, shortpath)

        logger.info("cross-validated results when optimizing for '%s':", self.config["optimize"])
        for metric, score in sorted(best_results["score"].items()):
            logger.info("%15s: %0.4f", metric, score)

        return best_results
