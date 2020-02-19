import random
import torch.utils.data

from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class TrainDataset(torch.utils.data.IterableDataset):
    """
    Samples training data. Intended to be used with a pytorch DataLoader
    """

    def __init__(self, search_run, benchmark, extractor):
        self.search_run = search_run
        self.benchmark = benchmark
        self.extractor = extractor

    def __iter__(self):
        """
        Returns: Triplets of the form (query_feature, posdoc_feature, negdoc_feature)
        """
        extractor = self.extractor
        benchmark = self.benchmark
        search_run = self.search_run

        train_qids = benchmark.folds[benchmark.cfg["fold"]]["train_qids"]
        is_rundocsonly = benchmark.cfg["rundocsonly"]
        if is_rundocsonly:
            # Create qrels from only those docs that showed up in the search results
            qrels = {
                qid: {docid: label for docid, label in benchmark.qrels[qid].items() if docid in search_run[qid]}
                for qid in train_qids
            }
        else:
            qrels = benchmark.qrels

        qid_to_reldocs = {qid: [docid for docid, label in qrels[qid].items() if label > 0] for qid in qrels}
        qid_to_negdocs = {qid: [docid for docid, label in qrels[qid].items() if label <= 0] for qid in qrels}

        # Remove the qids that does not have relevant/irrelevant docs associated with it
        valid_qids = [qid for qid in train_qids if qid_to_reldocs.get(qid) and qid_to_negdocs.get(qid)]

        # Convert each query and doc id to the corresponding feature/embedding and yield
        def genf():
            while True:
                # random.shuffle(valid_qids)

                for qid in valid_qids:
                    posdocid = random.choice(qid_to_reldocs[qid])
                    negdocid = random.choice(qid_to_negdocs[qid])

                    try:
                        query_feature, posdoc_feature, negdoc_feature = extractor.id2vec(qid, posdocid, negdocid)
                        yield {"query": query_feature, "posdoc": posdoc_feature, "negdoc": negdoc_feature}
                    # TODO: Replace below catch-all exception with MissingDocError
                    except Exception:
                        # The extractor should emit the above error
                        logger.warning("got none features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid)
                        raise

        return genf()


class PredDataset(torch.utils.data.IterableDataset):
    """
    Creates a Dataset for evaluation (test) data to be used with a pytorch DataLoader
    """

    def __init__(self, search_run, benchmark, extractor):
        self.search_run = search_run
        self.benchmark = benchmark
        self.extractor = extractor

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """
        benchmark = self.benchmark
        extractor = self.extractor
        search_run = self.search_run
        test_qids = benchmark.folds[benchmark.cfg["fold"]]["predict"]["test"]

        def genf():
            while True:
                for qid in test_qids:
                    for posdocid in search_run[qid].keys():
                        try:
                            query_feature, posdoc_feature = extractor.id2vec(qid, posdocid)
                            yield {"query": query_feature, "posdoc": posdoc_feature}
                        # TODO: Replace with MissingDocError
                        except Exception:
                            logger.warning("got none features: qid=%s posid=%s", qid, posdocid)
                            raise

        return genf()
