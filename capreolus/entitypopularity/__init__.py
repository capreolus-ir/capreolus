from capreolus.registry import ModuleBase, RegisterableModule, Dependency

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

class EntityPopularity(ModuleBase, metaclass=RegisterableModule):
    "the module base class"

    module_type = "entitypopularity"


class EntityPopularityCentralityDegree(EntityPopularity):
    name = "centralitydegree"

    dependencies = {
        'utils': Dependency(module="entityutils", name="wikilinks")
    }

    @staticmethod
    def config():
        direction = "in"

        if direction not in ["in", "out"]:
            raise ValueError(f"invalid direction (in/out)")

    def initialize(self):
        self["utils"].load_wp_links()

    def get_popularity_degree(self, e):
        return self.degree_centrality(e)

    def degree_centrality(self, e):
        e = "<{}>".format(e).replace(" ", "_")
        if self.cfg['direction'] == "in":
            nu = len(self['utils'].get_inlinks(e))
        else:
            nu = len(self['utils'].get_outlinks(e))
        de = self['utils'].total_nodes_count - 1
        return nu / de
