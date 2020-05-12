import random
from itertools import product
import hashlib
import torch.utils.data

from capreolus.utils.exceptions import MissingDocError
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class TrainDataset(torch.utils.data.IterableDataset):
    """
    Samples training data. Intended to be used with a pytorch DataLoader
    """

    def __init__(self, qid_docid_to_rank, qrels, extractor):
        self.extractor = extractor
        self.iterations = 0  # TODO remove

        # remove qids from qid_docid_to_rank that do not have relevance labels in the qrels
        qid_docid_to_rank = qid_docid_to_rank.copy()
        for qid in list(qid_docid_to_rank.keys()):
            if qid not in qrels:
                logger.warning("skipping qid=%s that was missing from the qrels", qid)
                del qid_docid_to_rank[qid]

        self.qid_docid_to_rank = qid_docid_to_rank
        self.qid_to_reldocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) > 0] for qid, docids in qid_docid_to_rank.items()
        }

        self.qid_to_negdocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) <= 0] for qid, docids in qid_docid_to_rank.items()
        }

        # remove any ids that do not have both relevant and non-relevant documents for training
        total_samples = 1  # keep tracks of the total possible number of unique training triples for this dataset
        for qid in qid_docid_to_rank:
            posdocs = len(self.qid_to_reldocs[qid])
            negdocs = len(self.qid_to_negdocs[qid])
            total_samples += posdocs * negdocs
            if posdocs == 0 or negdocs == 0:
                logger.warning("removing training qid=%s with %s positive docs and %s negative docs", qid, posdocs, negdocs)
                del self.qid_to_reldocs[qid]
                del self.qid_to_negdocs[qid]

        self.total_samples = total_samples

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_docid_to_rank.items()])
        key_content = "{0}{1}".format(self.extractor.name, str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "train_{0}".format(key)

    def get_total_samples(self):
        return self.total_samples

    def generator_func(self):
        # Convert each query and doc id to the corresponding feature/embedding and yield
        while True:
            all_qids = sorted(self.qid_to_reldocs)
            if len(all_qids) == 0:
                raise RuntimeError("TrainDataset has no valid qids")

            random.shuffle(all_qids)

            for qid in all_qids:
                posdocid = random.choice(self.qid_to_reldocs[qid])
                negdocid = random.choice(self.qid_to_negdocs[qid])

                try:
                    yield self.extractor.id2vec(qid, posdocid, negdocid)
                except MissingDocError:
                    # at training time we warn but ignore on missing docs
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )

    def epoch_generator_func(self):
        """
        Generates all unique training triplets for the dataset
        """
        all_qids = sorted(self.qid_to_reldocs)
        for qid in all_qids:
            for posdoc_id in self.qid_to_reldocs[qid]:
                for negdoc_id in self.qid_to_negdocs[qid]:
                    try:
                        yield self.extractor.id2vec(qid, posdoc_id, negdoc_id)
                    except MissingDocError:
                        logger.warning(
                            "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdoc_id, negdoc_id
                        )

    def __iter__(self):
        """
        Returns: Triplets of the form (query_feature, posdoc_feature, negdoc_feature)
        """

        return iter(self.generator_func())


class PredDataset(torch.utils.data.IterableDataset):
    """
    Creates a Dataset for evaluation (test) data to be used with a pytorch DataLoader
    """

    def __init__(self, qid_docid_to_rank, extractor):
        self.qid_docid_to_rank = qid_docid_to_rank

        self.extractor = extractor
        def genf():
            for qid, docids in qid_docid_to_rank.items():
                for docid in docids:
                    try:
                        yield extractor.id2vec(qid, docid)
                    except MissingDocError:
                        # when predictiong we raise an exception on missing docs, as this may invalidate results
                        logger.error("got none features for prediction: qid=%s posid=%s", qid, docid)
                        raise

        self.generator_func = genf

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_docid_to_rank.items()])
        key_content = "{0}{1}".format(self.extractor.name, str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        return "dev_{0}".format(key)

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return iter(self.generator_func())

    def get_qid_docid_pairs(self):
        """
        Returns a generator for the (qid, docid) pairs. Useful if you want to sequentially access the pred pairs without
        extracting the actual content
        """
        return (pair for k, v in zip(self.qid_docid_to_rank.keys(), self.qid_docid_to_rank.values()) for pair in product([k], v))
