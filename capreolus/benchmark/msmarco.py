import os
import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_trec_topics
from capreolus.utils.loginit import get_logger

from . import Benchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


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

        for set_name in folds:
            cur_queriesfn = [fn for fn in queries_fn if f".{set_name}." in fn]
            cur_qrelfn = [fn for fn in qrels_fn if f".{set_name}." in fn]
            with open(gz_dir / cur_queriesfn[0], "r") as f:
                for line in f:
                    qid, query = line.strip().split("\t")
                    topic_f.write(topic_to_trectxt(qid, query))
                    folds[set_name].add(qid)

            if not cur_qrelfn:
                logger.warning(
                    f"{set_name} qrel is unfound. This is expected if it is eval set. "
                    f"This is unexpected if it is train or dev set."
                )
                continue

            with open(gz_dir / cur_qrelfn[0], "r") as f:
                for line in f:
                    qrel_f.write(line)

        # fold
        folds = {k: list(v) for k, v in folds.items()}
        folds = {"s1": {"train_qids": folds["train"], "predict": {"dev": folds["dev"], "test": folds["eval"]}}}
        json.dump(folds, open(self.fold_file, "w"))


@Benchmark.register
class MSMARCO_V2(Benchmark):
    module_name = "ms_v2"
    query_type = "title"

    dependencies = [Dependency(key="collection", module="collection")]  
    # could depends on 
    # dependencies = [Dependency(key="collection", module="collection", name="ms_v2"),]
    # config_spec = [ConfigOption("datasettype", "doc", "doc or pass, indicating which")]
    use_train_as_dev = False

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            qid_topic = [line.strip().split("\t") for line in open(self.topic_file)]
            self._topics = {
                self.query_type: {qid: topic for qid, topic in qid_topic},
            }
        return self._topics
    
    @property
    def dataset_type(self):
        if self.collection.module_name == "msdoc_v2":
            return "doc"
        elif self.collection.module_name == "mspsg_v2":
            return "pass"
        else:
            raise ValueError(f"Unexpected collection dependency: got {type} but expected 'doc' or 'pass'")

    def build(self):
        # type = self.config["datasettype"]
        self.data_dir = self.collection.data_dir
        self.qrel_file = self.data_dir / "qrels.txt"
        self.topic_file = self.data_dir / "topics.txt"
        self.fold_file = self.data_dir / "folds.json"
        self.download_if_missing()

    def download_if_missing(self):
        """ 
        This function only prepare folds.json from the existing topic files,  
        and assume both topic and qrels file are existing under the self.data_dir;

        """ 
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        assert all([f.exists() for f in [self.qrel_file, self.topic_file]])
        def load_qid_from_topic_tsv(topic_fn):
            return [line.strip().split("\t")[0] for line in open(topic_fn)]

        logger.info("preparing fold.json")

        self.data_dir.mkdir(exist_ok=True, parents=True)

        train_qids = load_qid_from_topic_tsv(self.data_dir / f"{self.dataset_type}v2_train_queries.tsv")
        dev_qids = load_qid_from_topic_tsv(self.data_dir / f"{self.dataset_type}v2_dev_queries.tsv")
        assert len(set(train_qids) & set(dev_qids)) == 0
        folds = {
            "s1": {
                "train_qids": train_qids, 
                "predict": {
                    "dev": dev_qids, 
                    "test": dev_qids, 
        }}}
        json.dump(folds, open(self.fold_file, "w"))
