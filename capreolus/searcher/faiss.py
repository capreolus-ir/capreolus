import torch
import pickle
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

    def query_from_file(self, topicsfn, output_path, fold=None):
        output_path = self._query_from_file(topicsfn, output_path, fold=fold)
        
        return output_path


    def _query_from_file(self, topicsfn, output_path, fold=None):
        assert fold is not None

        # `qid_query` contains (qid, query) tuples in the order they were encoded

        # A manual search is done over the docs in dev_run - this way for each qid, we only score the docids that BM25 retrieved for it
        topic_vectors, qid_query = self.create_topic_vectors_for_fold(topicsfn, output_path, fold)
        self.index.manual_search(topic_vectors, 100, qid_query, output_path, fold)
        distances, results = self.index.faiss_search(topic_vectors, 100, qid_query, fold)
        distances = distances.astype(np.float16)

        self.write_results_in_trec_format(results, distances, qid_query, output_path)

        return output_path

    def build_encoder(self, fold):
        """
        TODO: Deprecate this method. We should not rely on BM25 search results to load the pre-trained encoder.
        Solution: Do not use BM25 run to form the TrainTripletSampler while training the encoder
        """

        rank_results = self.index.evaluate_bm25_search()
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)
        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}
        dev_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["predict"]["dev"]}
        encoder_qids = best_search_run.keys()
        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)

        self.index.encoder.build_model(train_run, dev_run, docids, encoder_qids)

    def create_topic_vectors_for_fold(self, topicsfn, output_path, fold):
        """
        Creates a tensor of shape (num_queries, emb_size). Uses all the topics available in the dataset. Filtering based on folds is done later
        """
        self.build_encoder(fold)
        topics = load_trec_topics(topicsfn)
        # TODO: Use the test qids in the below line
        qids = sorted([qid for qid in self.benchmark.folds[fold]["predict"]["dev"]])
        qid_query = sorted([(qid, topics["title"][qid]) for qid in qids])
        tokenizer = self.index.encoder.extractor.tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        topic_vectors = []
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
        faiss_id_to_doc_id = pickle.load(open("faiss_id_to_doc_id.dump", "rb"))
        trec_string = "{qid} 0 {doc_id} {rank} {score} faiss\n"
        num_queries, num_neighbours = results.shape
        assert num_queries == len(qid_query)

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "faiss.run"), "w") as f:
            for i in range(num_queries):
                faiss_ids = results[i][results[i] > -1]
                qid = qid_query[i][0]

                for j, faiss_id in enumerate(faiss_ids):
                    doc_id = faiss_id_to_doc_id[faiss_id]
                    f.write(trec_string.format(qid=qid, doc_id=doc_id, rank=j+1, score=distances[i][j]))

        faiss_logger.debug("The search results in TREC format are at: {}".format(output_path))

        return output_path
