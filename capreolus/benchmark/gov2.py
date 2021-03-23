from capreolus import Dependency, constants
from . import Benchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class GOV2(Benchmark):
    module_name = "gov2"
    dependencies = [Dependency(key="collection", module="collection", name="gov2")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.gov2.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.gov2.txt"
    fold_file = PACKAGE_PATH / "data" / "gov2.folds.json"
    query_type = "title"


@Benchmark.register
class MQ2007(Benchmark):
    module_name = "mq2007"
    dependencies = [Dependency(key="collection", module="collection", name="gov2")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.mq2007.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.mq2007.txt"
    fold_file = PACKAGE_PATH / "data" / "mq2007.folds.json"
    query_type = "title"
