from capreolus import constants

PACKAGE_PATH = constants["PACKAGE_PATH"]


# TODO add download_if_missing and re-enable
# @Benchmark.register
# class MSMarcoPassage(Benchmark):
#     module_name = "msmarcopassage"
#     dependencies = [Dependency(key="collection", module="collection", name="msmarco")]
#     qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcopassage.txt"
#     topic_file = PACKAGE_PATH / "data" / "topics.msmarcopassage.txt"
#     fold_file = PACKAGE_PATH / "data" / "msmarcopassage.folds.json"
#     query_type = "title"
