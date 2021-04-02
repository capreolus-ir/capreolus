from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Core17(IRDBenchmark):
    module_name = "core17"
    query_type = "title"
    ird_dataset_names = ["nyt/trec-core-2017"]
    dependencies = [Dependency(key="collection", module="collection", name="nyt")]
    fold_file = PACKAGE_PATH / "data" / "core17_birch_folds.json"


@Benchmark.register
class Core17Desc(Core17):
    module_name = "core17.desc"
    query_type = "desc"
    fold_file = PACKAGE_PATH / "data" / "core17_birch_folds.json"
