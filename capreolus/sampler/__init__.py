import random
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

        self.qid_to_reldocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) > 0]
            for qid, docids in qid_docid_to_rank.items()
        }

        self.qid_to_negdocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) <= 0]
            for qid, docids in qid_docid_to_rank.items()
        }

        # remove any qids that do not have both relevant and non-relevant documents for training
        for qid in qid_docid_to_rank:
            posdocs = len(self.qid_to_reldocs[qid])
            negdocs = len(self.qid_to_negdocs[qid])

            if posdocs == 0 or negdocs == 0:
                logger.warning(
                    "removing training qid=%s with %s positive docs and %s negative docs",
                    qid,
                    posdocs,
                    negdocs,
                )
                del self.qid_to_reldocs[qid]
                del self.qid_to_negdocs[qid]

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
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s",
                        qid,
                        posdocid,
                        negdocid,
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
        def genf():
            for qid, docids in qid_docid_to_rank.items():
                for docid in docids:
                    try:
                        yield extractor.id2vec(qid, docid)
                    except MissingDocError:
                        # when predictiong we raise an exception on missing docs, as this may invalidate results
                        logger.error(
                            "got none features for prediction: qid=%s posid=%s",
                            qid,
                            docid,
                        )
                        raise

        self.generator_func = genf

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return iter(self.generator_func())
