from capreolus import ConfigOption, Dependency, constants

from . import Benchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class DummyBenchmark(Benchmark):
    """ Tiny benchmark for testing """

    module_name = "dummy"
    dependencies = [Dependency(key="collection", module="collection", name="dummy")]
    config_spec = [ConfigOption("fold", "s1", "fold to run")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"
