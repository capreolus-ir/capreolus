from capreolus.registry import ModuleBase, RegisterableModule, Dependency

# from extractor import Extractor


class Reranker(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "reranker"
    dependencies = {"extractor": Dependency(module="extractor", name="embedtext")}

    cfg = {}


class KNRM(Reranker):
    name = "KNRM"
    # dependencies = {"extractor": "EmbedText"}

    @staticmethod
    def config():
        gradkernels = True
        scoretanh = False


class PACRR(Reranker):
    name = "PACRR"

    @staticmethod
    def config():
        kmax = 5
        nfilters = 2
