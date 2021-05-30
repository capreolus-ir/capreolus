from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Gov2(IRDBenchmark):
    module_name = "gov2"
    query_type = "title"
    ird_dataset_names = ["gov2/trec-tb-2004", "gov2/trec-tb-2005", "gov2/trec-tb-2006"]
    dependencies = [Dependency(key="collection", module="collection", name="gov2")]
    fold_file = PACKAGE_PATH / "data" / "gov2_maxp_folds.json"


@Benchmark.register
class Gov2Desc(Gov2):
    module_name = "gov2.desc"
    query_type = "desc"
