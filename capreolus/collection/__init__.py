from capreolus.registry import ModuleBase, RegisterableModule


class Collection(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "collection"
    is_large_collection = False

    def get_path_and_types(self):
        return self.path, self.collection_type, self.generator_type


class Robust04(Collection):
    name = "robust04"
    # path = "/home/andrew/Aquaint-TREC-3-4"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class Robust05(Collection):
    name = "robust05"
    path = "missingpath"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"
