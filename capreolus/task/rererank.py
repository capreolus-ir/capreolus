import os

import numpy as np
from profane import ConfigOption, Dependency

from capreolus import evaluator
from capreolus.sampler import PredDataset
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


@Task.register
class ReRerankTask(Task):
    module_name = "rererank"
    config_spec = [
        ConfigOption("fold", "s1", "fold to run"),
        ConfigOption("optimize", "map", "metric to maximize on the dev set"),
        ConfigOption("topn", 100, "number of stage two results to rerank"),
    ]
    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="robust04.yang19", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="rank", module="task", name="rank", provide_this=True),
        Dependency(key="rerank1", module="task", name="rerank"),
        Dependency(key="rerank2", module="task", name="rerank"),
    ]

    commands = ["train", "evaluate", "traineval"] + Task.help_commands
    default_command = "describe"

    def traineval(self):
        self.train()
        self.evaluate()

    def train(self):
        fold = self.config["fold"]
        logger.debug("results path: %s", self.get_results_path())

        self.rank.search()
        rank_results = self.rank.evaluate()
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        second_stage_results = self.rerank1.rerank_run(best_search_run, self.rerank1.get_results_path(), include_train=True)
        second_stage_topn = {
            qid: dict(sorted(docids.items(), key=lambda x: x[1], reverse=True)[: self.config["topn"]])
            for split in ("train", "dev", "test")
            for qid, docids in second_stage_results[split].items()
        }

        third_stage_results = self.rerank2.rerank_run(second_stage_topn, self.get_results_path())
        return third_stage_results

    def evaluate(self):
        fold = self.config["fold"]
        train_output_path = self.get_results_path()
        test_output_path = train_output_path / "pred" / "test" / "best"
        logger.debug("results path: %s", train_output_path)

        if os.path.exists(test_output_path):
            test_preds = Searcher.load_trec_run(test_output_path)
        else:
            self.rank.search()
            rank_results = self.rank.evaluate()
            best_search_run_path = rank_results["path"][fold]
            best_search_run = Searcher.load_trec_run(best_search_run_path)

            docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
            self.reranker.extractor.preprocess(
                qids=best_search_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type]
            )
            self.reranker.build_model()
            self.reranker.searcher_scores = best_search_run

            self.reranker.trainer.load_best_model(self.reranker, train_output_path)

            test_run = {
                qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["predict"]["test"]
            }
            test_dataset = PredDataset(qid_docid_to_rank=test_run, extractor=self.reranker.extractor)

            test_preds = self.reranker.trainer.predict(self.reranker, test_dataset, test_output_path)

        metrics = evaluator.eval_runs(test_preds, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)
        logger.info("rerank: fold=%s test metrics: %s", fold, metrics)

        print("\ncomputing metrics across all folds")
        avg = {}
        found = 0
        for fold in self.benchmark.folds:
            # TODO fix by using multiple Tasks
            from pathlib import Path

            pred_path = Path(test_output_path.as_posix().replace("fold-" + self.config["fold"], "fold-" + fold))
            if not os.path.exists(pred_path):
                print("\tfold=%s results are missing and will not be included" % fold)
                continue

            found += 1
            preds = Searcher.load_trec_run(pred_path)
            metrics = evaluator.eval_runs(preds, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)
            for metric, val in metrics.items():
                avg.setdefault(metric, []).append(val)

        avg = {k: np.mean(v) for k, v in avg.items()}
        logger.info("rerank: average cross-validated metrics when choosing iteration based on '%s':", self.config["optimize"])
        for metric, score in sorted(avg.items()):
            logger.info("%25s: %0.4f", metric, score)
