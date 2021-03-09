from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Genomics(IRDBenchmark):
    module_name = "genomics"
    query_type = "text"
    ird_dataset_names = ["highwire/trec-genomics-2006", "highwire/trec-genomics-2007"]
    dependencies = [Dependency(key="collection", module="collection", name="highwire")]
    fold_file = PACKAGE_PATH / "data" / "genomics_5folds.json"
