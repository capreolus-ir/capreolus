from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Core18(IRDBenchmark):
    module_name = "core18"
    query_type = "text"
    ird_dataset_names = ["wapo/v2/trec-core-2018"]
    dependencies = [Dependency(key="collection", module="collection", name="wapo")]
    fold_file = PACKAGE_PATH / "data" / "core18_title_folds.json"


@Benchmark.register
class Core18Desc(Core18):
    module_name = "core18.desc"
    fold_file = PACKAGE_PATH / "data" / "core18_desc_folds.json"
