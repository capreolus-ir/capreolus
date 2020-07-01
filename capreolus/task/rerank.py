import os
from pathlib import Path

from profane import ConfigOption, Dependency

from capreolus import evaluator
from capreolus.sampler import PredDataset, TrainDataset
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


@Task.register
class RerankTask(Task):
    module_name = "rerank"
    config_spec = [
        ConfigOption("fold", "s1", "fold to run"),
        ConfigOption("optimize", "map", "metric to maximize on the dev set"),  # affects train() because we check to save weights
    ]
    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="robust04.yang19", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="rank", module="task", name="rank"),
        Dependency(key="reranker", module="reranker", name="KNRM"),
    ]

    commands = ["train", "evaluate", "traineval"] + Task.help_commands
    default_command = "describe"

    def traineval(self):
        self.train()
        self.evaluate()

    def train(self):
        fold = self.config["fold"]

        self.rank.search()
        rank_results = self.rank.evaluate()
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        return self.rerank_run(best_search_run, self.get_results_path())

    def rerank_run(self, best_search_run, train_output_path, include_train=False):
        if not isinstance(train_output_path, Path):
            train_output_path = Path(train_output_path)

        fold = self.config["fold"]
        dev_output_path = train_output_path / "pred" / "dev"
        logger.debug("results path: %s", train_output_path)

        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
        self.reranker.extractor.preprocess(
            qids=best_search_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type]
        )
        self.reranker.build_model()
        self.reranker.searcher_scores = best_search_run

        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}
        dev_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["predict"]["dev"]}

        train_dataset = TrainDataset(
            qid_docid_to_rank=train_run,
            qrels=self.benchmark.qrels,
            extractor=self.reranker.extractor,
            relevance_level=self.benchmark.relevance_level,
        )
        dev_dataset = PredDataset(qid_docid_to_rank=dev_run, extractor=self.reranker.extractor)

        self.reranker.trainer.train(
            self.reranker,
            train_dataset,
            train_output_path,
            dev_dataset,
            dev_output_path,
            self.benchmark.qrels,
            self.config["optimize"],
            self.benchmark.relevance_level,
        )

        self.reranker.trainer.load_best_model(self.reranker, train_output_path)
        dev_output_path = train_output_path / "pred" / "dev" / "best"
        dev_preds = self.reranker.trainer.predict(self.reranker, dev_dataset, dev_output_path)

        test_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["predict"]["test"]}
        test_dataset = PredDataset(qid_docid_to_rank=test_run, extractor=self.reranker.extractor)
        test_output_path = train_output_path / "pred" / "test" / "best"
        test_preds = self.reranker.trainer.predict(self.reranker, test_dataset, test_output_path)

        preds = {"dev": dev_preds, "test": test_preds}

        if include_train:
            train_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=self.reranker.extractor)
            train_output_path = train_output_path / "pred" / "train" / "best"
            train_preds = self.reranker.trainer.predict(self.reranker, train_dataset, train_output_path)
            preds["train"] = train_preds

        return preds

    def evaluate(self):
        fold = self.config["fold"]
        train_output_path = self.get_results_path()
        logger.debug("results path: %s", train_output_path)

        searcher_runs, reranker_runs = self.find_crossvalidated_results()

        if fold not in reranker_runs:
            logger.error("could not find predictions; run the train command first")
            raise ValueError("could not find predictions; run the train command first")

        fold_dev_metrics = evaluator.eval_runs(
            reranker_runs[fold]["dev"], self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s dev metrics: %s", fold, fold_dev_metrics)

        fold_test_metrics = evaluator.eval_runs(
            reranker_runs[fold]["test"], self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level
        )
        logger.info("rerank: fold=%s test metrics: %s", fold, fold_test_metrics)

        if len(reranker_runs) != len(self.benchmark.folds):
            logger.info(
                "rerank: skipping cross-validated metrics because results exist for only %s/%s folds",
                len(reranker_runs),
                len(self.benchmark.folds),
            )
            return {
                "fold_test_metrics": fold_test_metrics,
                "fold_dev_metrics": fold_dev_metrics,
                "cv_metrics": None,
                "interpolated_cv_metrics": None,
            }

        logger.info("rerank: average cross-validated metrics when choosing iteration based on '%s':", self.config["optimize"])
        all_preds = {}
        for preds in reranker_runs.values():
            for qid, docscores in preds["test"].items():
                all_preds.setdefault(qid, {})
                for docid, score in docscores.items():
                    all_preds[qid][docid] = score

        cv_metrics = evaluator.eval_runs(
            all_preds, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level
        )
        interpolated_results = evaluator.interpolated_eval(
            searcher_runs, reranker_runs, self.benchmark, self.config["optimize"], evaluator.DEFAULT_METRICS
        )

        for metric, score in sorted(cv_metrics.items()):
            logger.info("%25s: %0.4f", metric, score)

        logger.info("interpolated with alphas = %s", sorted(interpolated_results["alphas"].values()))
        for metric, score in sorted(interpolated_results["score"].items()):
            logger.info("%25s: %0.4f", metric + " [interp]", score)

        return {
            "fold_test_metrics": fold_test_metrics,
            "fold_dev_metrics": fold_dev_metrics,
            "cv_metrics": cv_metrics,
            "interpolated_results": interpolated_results,
        }

    def find_crossvalidated_results(self):
        searcher_runs = {}
        rank_results = self.rank.evaluate()
        for fold in self.benchmark.folds:
            searcher_runs[fold] = {"dev": Searcher.load_trec_run(rank_results["path"][fold])}
            searcher_runs[fold]["test"] = searcher_runs[fold]["dev"]

        reranker_runs = {}
        train_output_path = self.get_results_path()
        test_output_path = train_output_path / "pred" / "test" / "best"
        dev_output_path = train_output_path / "pred" / "dev" / "best"
        for fold in self.benchmark.folds:
            # TODO fix by using multiple Tasks
            test_path = Path(test_output_path.as_posix().replace("fold-" + self.config["fold"], "fold-" + fold))
            if os.path.exists(test_path):
                reranker_runs.setdefault(fold, {})["test"] = Searcher.load_trec_run(test_path)

                dev_path = Path(dev_output_path.as_posix().replace("fold-" + self.config["fold"], "fold-" + fold))
                reranker_runs.setdefault(fold, {})["dev"] = Searcher.load_trec_run(dev_path)

        return searcher_runs, reranker_runs
