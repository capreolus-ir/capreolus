import torch
import os
import numpy as np
from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.trec import load_trec_topics
from capreolus import get_logger

from . import Searcher


logger = get_logger(__name__)
faiss_logger = get_logger("faiss")


@Searcher.register
class FAISSSearcher(Searcher):
    """
    Use FAISS to do approximate nearest-neighbor search on embeddings created using sentenceBERT
    """

    module_name = "faiss"

    dependencies = [Dependency(key="index", module="index", name="faiss"), Dependency(key="benchmark", module="benchmark")]

    def _query_from_file(self, topicsfn, output_path, config):
        param_str = ""
        
        # `qid_query` contains (qid, query) tuples in the order they were encoded
        topic_vectors, qid_query = self.create_topic_vectors(topicsfn, output_path)

        distances, results = self.index.search(topic_vectors, 100)

        return self.write_results_in_trec_format(results, distances, qid_query, output_path)

    def create_topic_vectors(self, topicsfn, output_path):
        self.index.encoder.build_model()
        topics = load_trec_topics(topicsfn)
        topic_vectors = []

        qid_query = sorted([(qid, query) for qid, query in topics["title"].items() if qid in self.benchmark.folds["s1"]["train_qids"]])
        tokenizer = self.index.encoder.extractor.tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for qid, query in qid_query:
                query_toks = tokenizer.tokenize(query)[:510]
                numericalized_query = tokenizer.convert_tokens_to_ids(["[CLS]"] + query_toks + ["[SEP]"])
                numericalized_query = torch.tensor(numericalized_query).to(device)
                numericalized_query = numericalized_query.reshape(1, -1)
                topic_vector = self.index.encoder.encode(numericalized_query).cpu().numpy()
                topic_vectors.append(topic_vector)
                
        return np.concatenate(topic_vectors, axis=0), qid_query

    def write_results_in_trec_format(self, results, distances, qid_query, output_path):
        trec_string = "{qid} 0 {doc_id} {rank} {score} faiss\n"
        num_queries, num_neighbours = results.shape
        assert num_queries == len(qid_query)

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "faiss.run"), "w") as f:
            for i in range(num_queries):
                lucene_doc_ids = results[i][results[i] > -1]
                doc_ids = self.index.index.convert_lucene_ids_to_doc_ids(lucene_doc_ids)
                qid = qid_query[i][0]

                for j, doc_id in enumerate(doc_ids):
                    f.write(trec_string.format(qid=qid, doc_id=doc_id, rank=j+1, score=distances[i][j]))

        faiss_logger.debug("The search results in TREC format are at: {}".format(output_path))
        return output_path
                
            


