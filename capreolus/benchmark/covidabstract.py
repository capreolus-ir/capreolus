from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class CovidAbstract(IRDBenchmark):
    module_name = "covidabstract"
    query_type = "title"
    ird_dataset_names = ["cord19/trec-covid"]
    dependencies = [Dependency(key="collection", module="collection", name="covidabstract")]
    fold_file = PACKAGE_PATH / "data" / "covid_random_folds.json"
