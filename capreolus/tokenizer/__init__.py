from capreolus.registry import ModuleBase, RegisterableModule, Dependency


class Tokenizer(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "tokenizer"


class AnseriniTokenizer(Tokenizer):
    name = "anserini"

    @staticmethod
    def config():
        keepstops = True
        stemmer = "none"
