from capreolus.registry import ModuleBase, RegisterableModule, Dependency


class Extractor(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "extractor"


class EmbedText(Extractor):
    name = "embedtext"

    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"keepstops": True}),
        "tokenizer": Dependency(module="tokenizer", name="anserini"),
    }

    @staticmethod
    def config():
        keepstops = False
