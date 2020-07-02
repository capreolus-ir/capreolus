from capreolus import Dependency, constants

from . import Benchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Robust04(Benchmark):
    """ Robust04 benchmark using the title folds from Huston and Croft. [1] Each of these is used as the test set.
        Given the remaining four folds, we split them into the same train and dev sets used in recent work. [2]

        [1] Samuel Huston and W. Bruce Croft. 2014. Parameters learned in the comparison of retrieval models using term dependencies. Technical Report.

        [2] Sean MacAvaney, Andrew Yates, Arman Cohan, Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. SIGIR 2019.
    """

    module_name = "robust04"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_cedr_folds.json"
    query_type = "title"


@Benchmark.register
class Robust04Yang19(Benchmark):
    """Robust04 benchmark using the folds from Yang et al. [1]

    [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. 2019. Critically Examining the "Neural Hype": Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    module_name = "robust04.yang19"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"
