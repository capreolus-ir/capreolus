from capreolus import ModuleBase, Dependency, ConfigOption, get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class Index(ModuleBase):
    module_type = "index"
    dependencies = [Dependency(key="collection", module="collection")]

    def get_index_path(self):
        return self.get_cache_path() / "index"

    def exists(self):
        donefn = self.get_index_path() / "done"
        return donefn.exists()

    def create_index(self):
        if self.exists():
            return

        self._create_index()
        donefn = self.get_index_path() / "done"
        with open(donefn, "wt") as donef:
            print("done", file=donef)

    def _create_index(self):
        raise NotImplementedError()

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def get_docs(self, doc_ids):
        raise NotImplementedError()


from profane import import_all_modules

from .anserini import AnseriniIndex

import_all_modules(__file__, __package__)
