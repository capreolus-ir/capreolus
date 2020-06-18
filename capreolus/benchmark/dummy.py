from . import Benchmark
from capreolus import constants, ConfigOption, Dependency

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class DummyBenchmark(Benchmark):
    module_name = "dummy"
    dependencies = [Dependency(key="collection", module="collection", name="dummy")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"
