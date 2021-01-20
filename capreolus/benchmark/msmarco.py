import os
import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_trec_topics
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

PACKAGE_PATH = constants["PACKAGE_PATH"]
from . import Benchmark


@Benchmark.register
class MSMarcoPassage(Benchmark):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="collection", module="collection", name="msmarcopsg")]

    query_type = "title"
    config_spec = []
    use_train_as_dev = False

    data_dir = PACKAGE_PATH / "data" / "msmarcopsg"
    qrel_file = data_dir / "qrels.txt"
    topic_file = data_dir / "topics.txt"
    fold_file = data_dir / "folds.json"

    # @property
    # def topics(self):
    #     if not hasattr(self, "_topics"):
    #         self._topics = {"title": {}, "desc": {}, "narr": {}}
    #
    #         with open(self.topic_file, "r") as f:
    #             for line in f:
    #                 qid, query = line.split("\t")
    #                 self._topics["title"][qid] = query
    #                 self._topics["desc"][qid] = query
    #                 self._topics["narr"][qid] = query
    #
    #     return self._topics

    def build(self):
        self.download_if_missing()

    def download_if_missing(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        def match_size(fn):
            if ".train." in fn:
                return True

            # if self.config["qrelsize"] == "small":
            if True:
                return ".small." in fn
            return ".small." not in fn

        gz_dir = self.collection.download_raw()
        queries_fn = [fn for fn in os.listdir(gz_dir) if "queries." in fn and match_size(fn)]
        qrels_fn = [fn for fn in os.listdir(gz_dir) if "qrels." in fn and match_size(fn)]  # note that qrel.test is not given

        # topic and qrel
        topic_f, qrel_f = open(self.topic_file, "w"), open(self.qrel_file, "w")
        folds = {"train": set(), "dev": set(), "eval": set()}

        # Copy over the qrels
        qids_in_qrels = set()

        for set_name in folds:
            cur_qrelfn = [fn for fn in qrels_fn if f".{set_name}." in fn]
            if not cur_qrelfn:
                logger.warning(
                    f"{set_name} qrel is unfound. This is expected if it is eval set. "
                    f"This is unexpected if it is train or dev set."
                )
                continue

            with open(gz_dir / cur_qrelfn[0], "r") as f:
                for line in f:
                    qid, _, docid, label = line.split("\t")
                    qrel_f.write(line)
                    qids_in_qrels.add(qid)

        # Biw copy over the topics and form folds
        for set_name in folds:
            cur_queriesfn = [fn for fn in queries_fn if f".{set_name}." in fn]
            with open(gz_dir / cur_queriesfn[0], "r") as f:
                for line in f:
                    qid, query = line.strip().split("\t")
                    # Performance hack. No point in dealing with queries for which we have no relevance judgements
                    if qid not in qids_in_qrels:
                        continue
                    topic_f.write(topic_to_trectxt(qid, query))
                    folds[set_name].add(qid)

        folds = {k: list(v) for k, v in folds.items()}
        folds = {"s1": {"train_qids": folds["train"], "predict": {"dev": folds["dev"], "test": folds["eval"]}}}
        json.dump(folds, open(self.fold_file, "w"))


@Benchmark.register
class MSMarcoPassageKeywords(MSMarcoPassage):
    module_name = "msmarcopsg_keywords"
    dependencies = MSMarcoPassage.dependencies + [
        Dependency(
            key="tokenizer",
            module="tokenizer",
            name="anserini",
            default_config_overrides={"keepstops": False, "stemmer": "none"},  # don't keepStops, don't stem
        ),
    ]

    topic_file = MSMarcoPassage.data_dir / "topics.keyword.txt"

    def download_if_missing(self):
        super().download_if_missing()
        full_topic_file = super().topic_file
        assert full_topic_file.exists()
        if self.topic_file.exists():
            return

        title = load_trec_topics(full_topic_file)["title"]
        with open(self.topic_file, "w") as f:
            for qid, full_topic in title:
                kw_topic = self.tokenizer.tokenize(full_topic)
                f.write(topic_to_trectxt(qid, kw_topic))
