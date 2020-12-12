from capreolus import ConfigOption, Dependency, constants

from . import Searcher


@Searcher.register
class FAISSSearcher(Searcher):
    """
    Use FAISS to do approximate nearest-neighbor search on embeddings created using sentenceBERT
    """

    module_name = "faiss"

    dependencies = [Dependency(key="index", module="index", name="faiss")]

    def _query_from_file(self, topicsfn, output_path, config):
        param_str = ""
        self._faiss_query_from_file(topicsfn, param_str, output_path, config["fields"])

        return output_path

    def _faiss_query_from_file(*args, **kwargs):
        raise NotImplementedError

