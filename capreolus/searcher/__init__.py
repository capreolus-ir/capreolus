from capreolus.registry import ModuleBase, RegisterableModule, Dependency


class Searcher(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "searcher"
    cfg = {}


class SDM(Searcher):
    """ a module impl """

    # idea is that we need a 2nd index (extidx) for something like calculating IDF on a larger corpus,
    # so extidx.collection should be different than the top level collection
    # ... but maybe we should not solve this in the initial attempt?
    # ... and when we do solve, something like this? Dependency(name="extidx", module="index", cls="anserini", bound="..collection..")
    # dependencies = {"index": ("index", "anserini"), "extidx": ("index", "anserini"), "extidx.collection": ("collection", "collection")}

    dependencies = {"index": Dependency(module="index", name="anserini")}

    name = "SDM"

    @staticmethod
    def config():
        ow = 0.5
        uw = 0.2
