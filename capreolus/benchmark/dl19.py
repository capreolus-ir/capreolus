from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class DL19(IRDBenchmark):
    module_name = "dl19"
    query_type = "text"
    ird_dataset_names = ["msmarco-passage/trec-dl-2019"]
    dependencies = [Dependency(key="collection", module="collection", name="dl19")]
    fold_file = PACKAGE_PATH / "data" / "dl19_folds.json"

