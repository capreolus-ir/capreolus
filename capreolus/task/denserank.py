import math
import shutil
import time
import os
from collections import defaultdict
from capreolus import ConfigOption, Dependency, evaluator
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.searcher import Searcher
from capreolus.utils.trec import load_qrels

logger = get_logger(__name__)  # pylint: disable=invalid-name
faiss_logger = get_logger("faiss")


@Task.register
class DenseRankTask(Task):
    module_name = "denserank"
    requires_random_seed = False
    config_spec = [
        ConfigOption("fold", None, "fold to run"),
        ConfigOption("optimize", "map", "metric to maximize on the dev set"),
        ConfigOption("metrics", "default", "metrics reported for evaluation", value_type="strlist"),
        ConfigOption("numshards", 1, "Number of shards to split the FAISS index to"),
        ConfigOption("shard", 1, "The current shard that's being processed. See the 'createshard' command"),
    ]
    config_keys_not_in_path = ["optimize", "metrics", "shard"]  # affect only evaluation but not search()

    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="robust04.yang19", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="searcher", module="searcher", name="BM25"),
        Dependency(key="annsearcher", module="searcher", name="faiss"),
        Dependency(key="encoder", module="encoder", name="repbertpretrained"),
    ]

    commands = ["run", "evaluate", "createshard", "trainencoder", "outputpath", "deleteresults"] + Task.help_commands
    default_command = "describe"

    def do_bm25_search(self, fold):
        topics_fn = self.benchmark.get_topics_file()
        output_dir = os.path.join(self.get_results_path(), "rank")

        if hasattr(self.searcher, "index"):
            # All anserini indexes ignore the "fold" parameter. This is required for FAISS though, since we have to train an encoder
            self.searcher.index.create_index(fold=fold)

        search_results_folder = self.searcher.query_from_file(topics_fn, output_dir, fold=fold)
        logger.info("searcher results written to: %s", search_results_folder)
        metrics = evaluator.DEFAULT_METRICS
        best_results = evaluator.search_best_run(
            search_results_folder, self.benchmark, primary_metric="map", metrics=metrics, folds=fold
        )
        best_search_run_path = best_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        return best_search_run

    def outputpath(self):
        print(str(self.get_results_path()))

    def trainencoder(self):
        encoder = self.encoder
        fold = self.config["fold"]

        bm25_run = self.do_bm25_search(fold)
        train_run = {qid: docs for qid, docs in bm25_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}

        dev_run = defaultdict(dict)
        for qid, docs in bm25_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"] and qid in self.benchmark.qrels:
                for idx, (docid, score) in enumerate(docs.items()):
                    # Abritrary number for dev set. Should have used 1000, but don't have all day.
                    if idx >= 300:
                        break
                    dev_run[qid][docid] = score

        qids = bm25_run.keys()
        docids = set(docid for querydocs in bm25_run.values() for docid in querydocs)
        encoder.build_model(train_run, dev_run, docids, qids, self.get_results_path())

    def createshard(self):
        assert os.path.isfile(os.path.join(self.get_results_path(), "weights")), "Saved encoder weights not found at {}".format(
            self.get_results_path()
        )

        shard_id = self.config["shard"] - 1  # shard_ids are 1-indexed
        start_time = time.time()

        index_reader = self.searcher.index.get_anserini_index_reader()
        num_docs = index_reader.maxDoc()
        docs_per_shard = math.ceil(num_docs / self.config["numshards"])
        docids_for_current_shard = [
            self.searcher.index.convert_lucene_id_to_doc_id(x)
            for x in range(shard_id * docs_per_shard, (shard_id + 1) * docs_per_shard)
        ]
        docids_for_current_shard = [x for x in docids_for_current_shard if x is not None]

        logger.info("Getting all docs took {}".format(time.time() - start_time))
        offset = shard_id * docs_per_shard
        self.encoder.trainer.load_trained_weights(self.encoder, self.get_results_path())
        self.annsearcher.index.create_shard(
            self.encoder, shard_id, offset, docids_for_current_shard, self.config["fold"], self.get_results_path()
        )

    def evaluate(self):
        fold = self.config["fold"]
        output_path = self.get_results_path()
        for shard_id in range(self.config["numshards"]):
            assert os.path.isfile(
                os.path.join(output_path, "shard_{}_faiss_{}.index".format(shard_id, fold))
            ), "Shard {} does not exist".format(shard_id)

        self.encoder.trainer.load_trained_weights(self.encoder, output_path)
        topics = self.benchmark.topics
        index_reader = self.searcher.index.get_anserini_index_reader()
        num_docs = index_reader.maxDoc()
        docs_per_shard = math.ceil(num_docs / self.config["numshards"])
        search_results_folder = self.annsearcher._query_from_file(
            self.encoder, topics, output_path, self.config["numshards"], docs_per_shard, fold=fold
        )

    def deleteresults(self):
        output_path = self.get_results_path()
        logger.info("The output path is: ")
        logger.info(output_path)
        shutil.rmtree(output_path)
