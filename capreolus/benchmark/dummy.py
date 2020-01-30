import json
import os

from capreolus.benchmark import Benchmark
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Benchmark.register
class DummyBenchmark(Benchmark):
    name = "dummy"
    query_type = "title"

    @staticmethod
    def config():
        searcher = "bm25grid"
        collection = "dummy"
        rundocsonly = False
        return locals().copy()  # ignored by sacred

    def build(self):
        self.folds = json.load(open(os.path.join(self.collection.basepath, "dummy_folds.json"), "rt"))
        self.create_and_store_train_and_pred_pairs(self.folds)
