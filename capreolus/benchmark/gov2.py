from capreolus import Dependency, constants
from . import Benchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class MQ2007(Benchmark):
    module_name = "mq2007"
    dependencies = [Dependency(key="collection", module="collection", name="gov2")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.mq2007.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.mq2007.txt"
    fold_file = PACKAGE_PATH / "data" / "mq2007.folds.json"
    query_type = "title"
