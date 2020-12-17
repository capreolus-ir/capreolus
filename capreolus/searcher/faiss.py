import torch
import numpy as np
from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.trec import load_trec_topics
from capreolus import get_logger

from . import Searcher


logger = get_logger(__name__)


@Searcher.register
class FAISSSearcher(Searcher):
    """
    Use FAISS to do approximate nearest-neighbor search on embeddings created using sentenceBERT
    """

    module_name = "faiss"

    dependencies = [Dependency(key="index", module="index", name="faiss")]

    def _query_from_file(self, topicsfn, output_path, config):
        param_str = ""
        topic_vectors = self.create_topic_vectors(topicsfn, output_path)
        logger.info("Topic vectors have shape {}".format(topic_vectors.shape))
        distances, results = self.index.search(topic_vectors, 100)

        return self.write_results_in_trec_format(results, output_path)

    def create_topic_vectors(self, topicsfn, output_path):
        topics = load_trec_topics(topicsfn)
        topic_vectors = []
        self.index.encoder.build_model()

        with torch.no_grad():
            for qid, query in topics["title"].items():
                topic_vector = self.index.encoder.encode(query)
                topic_vectors.append(topic_vector)
                
        return np.concatenate(topic_vectors, axis=0)

    def write_results_in_trec_format(self, results, output_path):
        raise NotImplementedError
