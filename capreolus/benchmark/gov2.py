from capreolus import Dependency, constants, ConfigOption

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
class Gov2Passages(IRDBenchmark):
    module_name = "gov2passages"
    query_type = "title"
    ird_dataset_names = ["gov2/trec-tb-2004", "gov2/trec-tb-2005", "gov2/trec-tb-2006"]
    dependencies = [Dependency(key="collection", module="collection", name="gov2passages")]
    config_spec = [ConfigOption("pool", "max", "Strategy used to aggregate passage level scores")]
    fold_file = PACKAGE_PATH / "data" / "gov2_maxp_folds.json"
    need_pooling = True


@Benchmark.register
class Gov2Desc(Gov2):
    module_name = "gov2.desc"
    query_type = "desc"


@Benchmark.register
class MQ2007Passages(Benchmark):
    module_name = "mq2007passages"
    dependencies = [Dependency(key="collection", module="collection", name="gov2passages")]
    config_spec = [ConfigOption("pool", "max", "Strategy used to aggregate passage level scores")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.mq2007.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.mq2007.txt"
    fold_file = PACKAGE_PATH / "data" / "mq2007.folds.json"
    query_type = "title"
    need_pooling = True
