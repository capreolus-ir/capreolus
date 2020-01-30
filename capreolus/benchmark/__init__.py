from collections import defaultdict
import random

from tqdm import tqdm

from capreolus.utils.common import register_component_module, import_component_modules
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Benchmark:
    ALL = {}

    def __init__(self, search_run, collection, pipeline_config):
        self.search_run = search_run
        self.collection = collection
        self.pipeline_config = pipeline_config
        self.extractor = None  # Need this to transform training tuples with ids into embedding features
        self.reranking_runs = {}

    def set_extractor(self, extractor):
        self.extractor = extractor

    def build(self):
        """ Initialization method for subclasses to override """
        raise NotImplementedError

    def create_and_store_train_and_pred_pairs(self, folds):
        """
        Based on runs, generate and store pairs of (q_id, doc_ids) for future use.
        """
        # the run selected to rerank for each fold
        self.reranking_runs = {}

        # train on only docs that show up in the searcher? (rather than all judged docs)
        self.train_pairs = self.collection.qrels
        if self.pipeline_config["rundocsonly"]:
            self.train_pairs = {}

        # predict on only the docs to rerank (not on all judged docs)
        self.pred_pairs = {}

        for name, d in folds.items():
            dev_qids = set(d["train_qids"]) | set(d["predict"]["dev"])
            test_qids = set(d["predict"]["test"])
            full_search_run = self.search_run.crossvalidated_ranking(dev_qids, test_qids, full_run=True)
            self.reranking_runs[name] = full_search_run
            search_run = {qid: docscores for qid, docscores in full_search_run.items() if qid in test_qids}

            for qid, docscores in search_run.items():
                self.pred_pairs.setdefault(qid, []).extend(docscores.keys())

            if self.pipeline_config["rundocsonly"]:
                for qid, docscores in self.search_run.crossvalidated_ranking(dev_qids, set(d["train_qids"])).items():
                    self.train_pairs.setdefault(qid, set()).update(docscores.keys())

    def get_features(self, d):
        d = self.extractor.transform_qid_posdocid_negdocid(d["qid"], d["posdocid"], d.get("negdocid"))
        return d

    def pred_tuples(self, pred_pairs):
        if self.pipeline_config["sample"] == "simple":
            return self.simple_pred_tuples(pred_pairs)

        return None

    def training_tuples(self, qids):
        if self.pipeline_config["sample"] == "simple":
            return self.simple_training_tuples(qids)

        return None

    def simple_pred_tuples(self, pred_pairs):
        def predgenf():
            batch = defaultdict(list)
            for qid in tqdm(pred_pairs):
                for posdocid in pred_pairs[qid]:
                    features = self.get_features({"qid": qid, "posdocid": posdocid})
                    if features is None:
                        logger.warning("predict got none features: qid=%s docid=%s", qid, posdocid)
                        continue

                    for k, v in features.items():
                        batch[k].append(v)

                    if len(batch["qid"]) == self.pipeline_config["batch"]:
                        yield self.prepare_batch(batch)
                        batch = defaultdict(list)

            if len(batch["qid"]) > 0:
                missing = self.pipeline_config["batch"] - len(batch["qid"])
                for k in batch:
                    batch[k] = batch[k] + ([batch[k][-1]] * missing)
                yield self.prepare_batch(batch)

        logger.debug("Starting to get {0} pred pairs".format(len(pred_pairs)))
        x = list(predgenf())
        logger.debug("Done getting pred pairs")
        return x

    def get_posdocs_and_negdocs_for_qids(self, qids):
        labels = {
            qid: {docid: label for docid, label in self.collection.qrels[qid].items() if docid in self.train_pairs[qid]}
            for qid in qids
        }

        reldocs = {qid: [docid for docid, label in labels[qid].items() if label > 0] for qid in labels}
        negdocs = {qid: [docid for docid, label in labels[qid].items() if label <= 0] for qid in labels}

        return reldocs, negdocs

    def simple_training_tuples(self, qids):
        qid_order = [qid for qid in qids if qid in self.train_pairs and qid in self.collection.qrels]

        reldocs, negdocs = self.get_posdocs_and_negdocs_for_qids(qid_order)
        for qid in list(qid_order):
            if len(reldocs.get(qid, [])) == 0 or len(negdocs.get(qid, [])) == 0:
                qid_order.remove(qid)
                logger.warning("skipping qid=%s with no positive and/or negative samples", qid)

        def genf():
            batch = defaultdict(list)
            while True:
                random.shuffle(qid_order)
                for qid in qid_order:
                    posdocid = random.choice(reldocs[qid])
                    negdocid = random.choice(negdocs[qid])

                    features = self.get_features({"qid": qid, "posdocid": posdocid, "negdocid": negdocid})
                    if features is None:
                        logger.warning("got none features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid)
                        continue

                    for k, v in features.items():
                        batch[k].append(v)

                    if len(batch["qid"]) == self.pipeline_config["batch"]:
                        yield self.prepare_batch(batch)
                        batch = defaultdict(list)

        return genf()

    def prepare_batch(self, batch):
        return batch

    @staticmethod
    def config():
        raise NotImplementedError("config method must be provided by subclass")

    @classmethod
    def register(cls, subcls):
        return register_component_module(cls, subcls)


import_component_modules("benchmark")
