import random
import os


import numpy as np
import torch
from profane import ModuleBase, Dependency, ConfigOption, constants

from capreolus.sampler import TrainDataset, PredDataset
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus import evaluator
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
        Dependency(key="benchmark", module="benchmark", name="wsdm20demo", provide_this=True, provide_children=["collection"]),
        Dependency(key="rank", module="task", name="rank"),
        Dependency(key="reranker", module="reranker", name="KNRM"),
    ]

    commands = ["train", "evaluate", "traineval"] + Task.help_commands
    default_command = "describe"

    def traineval(self):
        self.train()
        self.evaluate()

    def train(self):
        torch.manual_seed(self.config["seed"])  # TODO move
        torch.cuda.manual_seed_all(self.config["seed"])

        fold = self.config["fold"]
        train_output_path = self.get_results_path()
        dev_output_path = train_output_path / "pred" / "dev"
        logger.debug("results path: %s", train_output_path)

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

        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}
        dev_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["predict"]["dev"]}

        train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=self.benchmark.qrels, extractor=self.reranker.extractor)
        dev_dataset = PredDataset(qid_docid_to_rank=dev_run, extractor=self.reranker.extractor)

        self.reranker.trainer.train(
            self.reranker,
            train_dataset,
            train_output_path,
            dev_dataset,
            dev_output_path,
            self.benchmark.qrels,
            self.config["optimize"],
        )

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

        metrics = evaluator.eval_runs(test_preds, self.benchmark.qrels, evaluator.DEFAULT_METRICS)
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
            metrics = evaluator.eval_runs(preds, self.benchmark.qrels, evaluator.DEFAULT_METRICS)
            for metric, val in metrics.items():
                avg.setdefault(metric, []).append(val)

        avg = {k: np.mean(v) for k, v in avg.items()}
        logger.info("rerank: average cross-validated metrics when choosing iteration based on '%s':", self.config["optimize"])
        for metric, score in sorted(avg.items()):
            logger.info("%15s: %0.4f", metric, score)
