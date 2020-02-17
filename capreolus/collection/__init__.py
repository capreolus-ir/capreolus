from capreolus.registry import ModuleBase, RegisterableModule


class Collection(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "collection"
    is_large_collection = False

    def get_path_and_types(self):
        return self.path, self.collection_type, self.generator_type

    def get_topics_path_and_type(self):
        return self.topic_path, self.topic_type

class Robust04(Collection):
    name = "robust04"
    # path = "/home/andrew/Aquaint-TREC-3-4"
    path = "/tuna1/collections/newswire/disk45"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    topic_path = "/home/x978zhan/mpi-spring/data/robust04/topics.robust04.301-450.601-700.txt"
    topic_type = "trec"

class Robust05(Collection):
    name = "robust05"
    path = "missingpath"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    topic_path = "missingpath"
    topic_type = "missingtype"
