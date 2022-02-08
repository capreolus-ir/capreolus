from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark, validate
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_trec_topics
import json
import os
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class DL19(IRDBenchmark):
    module_name = "dl19"
    query_type = "text"
    ird_dataset_names = ["msmarco-passage/trec-dl-2019"]
    dependencies = [Dependency(key="collection", module="collection", name="msmarcopsg")]

    # File paths are same as MSMarcoPassage
    data_dir = PACKAGE_PATH / "data" / "msmarcopsg"
    qrel_file = data_dir / "qrels.txt"
    topic_file = data_dir / "topics.txt"


    fold_file = PACKAGE_PATH / "data" / "dl19_folds.json"

    @validate
    def build(self):
        self.generate_folds()
    
    def generate_folds(self):

        def match_size(fn):
            # return True if the file is from training set, or the small version of dev and test set
            return (".train." in fn) or (".small." in fn)


        # topic and qrel
        folds = {"train": set(), "dev": set(), "eval": set()}

        # Add the DL19 qids (from test set)
        for qid in self.qrels.keys():
            folds['eval'].add(qid)

        self.data_dir.mkdir(exist_ok=True, parents=True)
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        gz_dir = self.collection.download_raw()
        queries_fn = [fn for fn in os.listdir(gz_dir) if "queries." in fn and match_size(fn)]
        qrels_fn = [fn for fn in os.listdir(gz_dir) if "qrels." in fn and match_size(fn)]  # note that qrel.test is not given

        # topic and qrel
        topic_f, qrel_f = open(self.topic_file, "w"), open(self.qrel_file, "w")

        for set_name in ["train", "dev"]:
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

