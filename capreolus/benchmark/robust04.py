import json
import os

from capreolus.collection import COLLECTIONS

from capreolus.benchmark import Benchmark
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Benchmark.register
class Robust04Benchmark(Benchmark):
    """ Benchmark using title queries with Robust2004 folds from [1] (Table 1) with the same dev (validation) and test folds as in [2]. That is, fold "sn" is the split whose test fold contains the query ids in fold n from [1].

        [1] Samuel Huston and W. Bruce Croft. Parameters learned in the comparison of retrieval models using term dependencies. Technical Report (2014).
        [2] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. CEDR: Contextualized Embeddings for Document Ranking. SIGIR 2019.
 """

    name = "robust04.title"
    query_type = "title"

    @staticmethod
    def config():
        fold = "s1"
        searcher = "bm25"
        collection = "robust04"
        rundocsonly = True  # use only docs from the searcher as pos/neg training instances (i.e., not all qrels)
        return locals().copy()  # ignored by sacred

    def build(self):
        self.folds = json.load(open(os.path.join(self.collection.basepath, "rob04_cedr_folds.json"), "rt"))
        self.create_and_store_train_and_pred_pairs(self.folds)


@Benchmark.register
class DemoRobust04Benchmark(Benchmark):
    """ Benchmark using title queries with Robust2004 folds and pipeline defaults corresponding to those used in [1].
        See the WSDM20 runbook for config options to use with each reranker.

        [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
"""

    name = "robust04.title.wsdm20demo"
    query_type = "title"

    @staticmethod
    def config():
        fold = "s1"
        searcher = "bm25staticrob04yang19"
        collection = "robust04"
        rundocsonly = True  # use only docs from the searcher as pos/neg training instances (i.e., not all qrels)
        maxqlen = 4
        maxdoclen = 800
        niters = 50
        batch = 32
        lr = 0.001
        softmaxloss = False

        stemmer = "none"
        indexstops = False
        return locals().copy()  # ignored by sacred

    def build(self):
        self.folds = json.load(open(os.path.join(self.collection.basepath, "rob04_yang19_folds.json"), "rt"))
        self.create_and_store_train_and_pred_pairs(self.folds)
