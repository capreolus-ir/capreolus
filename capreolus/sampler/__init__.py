from collections import defaultdict
from random import random

from capreolus.registry import ModuleBase, RegisterableModule, Dependency
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class Sampler(ModuleBase, metaclass=RegisterableModule):
    """
    Samples training and prediction data from the dataset
    """

    module_type = "sampler"
    dependencies = {"collection": Dependency(module="benchmark"), "extractor": Dependency(module="extractor")}

    @staticmethod
    def config():
        sampler = "uniform"  # Unused now. Will it ever make sense to do gaussian sampling?
        batch = 32

    def sample_training_data(self, search_run):
        """
        Samples training data from the collection, according to the information specified in the benchmark

        Args:
            search_run: The contents of a run file. The run file usually corresponds to a bm25 grid search run that
            produced the best results. The contents of search_fun is populated through Searcher.load_trec_run()

        Returns: A quadruple - (qid, query_text_embedding, posdoc_embedding, negdoc_embedding)

        """
        benchmark = self.modules["benchmark"]
        extractor = self.modules["extractor"]
        train_qids = benchmark.folds[benchmark.fold]["train_qids"]

        is_rundocsonly = benchmark.rundocsonly
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

        def genf():
            batch = defaultdict(list)
            while True:
                random.shuffle(valid_qids)

                for qid in valid_qids:
                    posdocid = random.choice(qid_to_reldocs[qid])
                    negdocid = random.choice(qid_to_negdocs[qid])

                    try:
                        query_feature = extractor.transform_text(benchmark.topics[qid])
                        posdoc_feature = extractor.transform_doc(posdocid)
                        negdoc_feature = extractor.transform_doc(negdocid)
                        batch["query"].append(query_feature)
                        batch["posdoc"].append(posdoc_feature)
                        batch["negdoc"].append(negdoc_feature)
                        if len(batch["query"]) == self.batch:
                            yield batch
                            batch = defaultdict(list)
                    except MissingDocError:
                        # The extractor should emit the above error
                        logger.warning("got none features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid)

        return genf()

    def sample_pred_data(self, search_run):
        """
        Samples prediction data from the collection, according to the information specified in the benchmark
        Args:
            search_run: 

        Returns:

        """
        benchmark = self.modules["benchmark"]
        extractor = self.modules["extractor"]
        test_qids = benchmark.folds[benchmark.fold]["test_qids"]

        def genf():
            batch = defaultdict(list)
            while True:
                for qid in test_qids:
                    for posdocid in search_run[qid].keys():
                        try:
                            query_feature = extractor.transform_text(benchmark.topics[qid])
                            posdoc_feature = extractor.transform_doc(posdocid)
                        except MissingDocError:
                            raise

                        batch["query"].append(query_feature)
                        batch["posdoc"].append(posdoc_feature)
                        if len(batch["query"]) == self.batch:
                            yield batch
                            batch = defaultdict(list)

                # If we have seen all qids but not enough to make a batch - just return whatever we have
                if len(batch["query"]) > 0:
                    yield batch

        return genf()
