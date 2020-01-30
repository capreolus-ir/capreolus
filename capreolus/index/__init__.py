import os
from capreolus.utils.common import register_component_module, import_component_modules


class Index:
    """ Module responsible for indexing a collection. """

    ALL = {}

    def __init__(self, collection, index_path, index_key):
        self.collection = collection
        self.index_path = index_path
        self.index_key = index_key

    @staticmethod
    def get_index_from_index_path(index_path):
        """
        Given an index path, we try to figure out the appropriate index class to use
        """
        for name, cls in Index.ALL.items():
            if name in index_path:
                return cls

        return None

    @staticmethod
    def config():
        raise NotImplementedError("config method must be provided by subclass")

    @classmethod
    def register(cls, subcls):
        return register_component_module(cls, subcls)

    def exists(self):
        return os.path.exists(os.path.join(self.index_path, "done"))

    def _build_index(self, config):
        raise NotImplementedError

    def create(self, config):
        if self.exists():
            return

        self._build_index(config)
        with open(os.path.join(self.index_path, "done"), "wt") as donef:
            print("done", file=donef)


import_component_modules("index")
