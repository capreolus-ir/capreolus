from capreolus import ConfigOption, Dependency, TrecRun, evaluator
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
        ConfigOption("allfolds", True, "run on each of the benchmark's folds"),
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
        if self.config["allfolds"]:
            # REF-TODO this is a dict, but normal output is a path ...
            return self.search_all_folds()

        search_results_folder = self.searcher.fit()
        logger.info("searcher results written to: %s", search_results_folder)
        return search_results_folder

        # REF-TODO handle docs to keep/remove logic
        # if self.config["filter"]:
        #     qrels = load_qrels(self.benchmark.qrel_ignore)
        #     docs_to_remove = {q: list(d.keys()) for q, d in qrels.items()}
        #     search_results_folder = self.searcher.query_from_file(topics_fn, output_dir, docs_to_remove)
        # else:
        #     search_results_folder = self.searcher.query_from_file(topics_fn, output_dir)

    def evaluate(self):
        if self.config["allfolds"]:
            return self.evaluate_all_folds()

        fit_path = self.searcher.fit_results
        if not fit_path:
            raise RuntimeError("searcher eval_path is not set")
        eval_path = self.searcher.query_from_benchmark()

        metrics = self.config["metrics"] if list(self.config["metrics"]) != ["default"] else evaluator.DEFAULT_METRICS
        best_results = evaluator.new_best_run(
            fit_path, eval_path, self.benchmark, primary_metric=self.config["optimize"], metrics=metrics
        )

        logger.info("rank: fold=%s best run: %s", self.benchmark.config["fold"], best_results["test_path"])
        for metric, score in sorted(best_results["score"].items()):
            logger.info("fold=%s %25s: %0.4f", self.benchmark.config["fold"], metric, score)

        return best_results

    def get_all_fold_tasks(self):
        single_fold_config = self.config.unfrozen_copy()
        single_fold_config["allfolds"] = False
        del single_fold_config["benchmark"]
        fold_tasks = [
            Task.create(self.module_name, config=single_fold_config, provide=fold_benchmark)
            for fold_benchmark in self.benchmark.get_all_fold_benchmarks()
        ]
        return fold_tasks

    def evaluate_all_folds(self):
        all_evals = {}
        all_results = TrecRun({})
        for task in self.get_all_fold_tasks():
            fold = task.benchmark.config["fold"]
            all_evals[fold] = task.evaluate()
            fold_results = TrecRun(all_evals[fold]["test_path"])
            all_results = all_results.union_qids(fold_results)

        metrics = self.config["metrics"] if list(self.config["metrics"]) != ["default"] else evaluator.DEFAULT_METRICS
        all_evals["score"] = all_results.evaluate(self.benchmark.qrels, metrics, self.benchmark.relevance_level)

        logger.info("rank: cross-validated results when optimizing for '%s':", self.config["optimize"])
        for metric, score in sorted(all_evals["score"].items()):
            logger.info("%25s: %0.4f", metric, score)

        return all_evals

    def search_all_folds(self):
        all_searches = {}
        for task in self.get_all_fold_tasks():
            fold = task.benchmark.config["fold"]
            all_searches[fold] = task.search()

        return all_searches
