from capreolus import Dependency, constants
from capreolus.utils.loginit import get_logger

from . import Benchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class ANTIQUE(Benchmark):
    """A Non-factoid Question Answering Benchmark from Hashemi et al. [1]

    [1] Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W. Bruce Croft. 2020. ANTIQUE: A non-factoid question answering benchmark. ECIR 2020.
    """

    module_name = "antique"
    dependencies = [Dependency(key="collection", module="collection", name="antique")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.antique.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.antique.txt"
    fold_file = PACKAGE_PATH / "data" / "antique.json"
    query_type = "title"
    relevance_level = 2
