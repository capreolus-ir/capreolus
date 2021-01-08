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

    def _query_from_file(self, topicsfn, output_path, config):
        param_str = ""
        
        # `qid_query` contains (qid, query) tuples in the order they were encoded
        topic_vectors, qid_query = self.create_topic_vectors(topicsfn, output_path)

        return self.index.manual_search(topic_vectors, 100, qid_query, output_path)
        # distances, results = self.index.search(topic_vectors, 100)
        # distances = distances.astype(np.float16)

        # return self.write_results_in_trec_format(results, distances, qid_query, output_path)

    def create_topic_vectors(self, topicsfn, output_path):
        rank_results = self.index.evaluate_bm25_search()
        best_search_run_path = rank_results["path"]["s1"]
        best_search_run = Searcher.load_trec_run(best_search_run_path)
        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds["s1"]["train_qids"]}
        dev_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds["s1"]["predict"]["dev"]}
        qids = best_search_run.keys()
        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)


        self.index.encoder.build_model(train_run, dev_run, docids, qids)
        topics = load_trec_topics(topicsfn)
        topic_vectors = []

        # qid_query = sorted([(qid, query) for qid, query in topics["title"].items() if qid in self.benchmark.folds["s1"]["predict"]["dev"]])
        qid_query = sorted([(qid, topics["title"][qid]) for qid in list(dev_run.keys())])
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
        
        # doc_304 = self.index.index.get_doc("FBIS3-20164")[:30]
        # doc_toks = tokenizer.tokenize(doc_304)[:510]
        # numericalized_doc = tokenizer.convert_tokens_to_ids(["[CLS]"] + doc_toks + ["[SEP]"])
        # numericalized_doc = torch.tensor(numericalized_doc).to(device)
        # numericalized_doc = numericalized_doc.reshape(1, -1)
        # with torch.no_grad():
            # encoded_doc = self.index.encoder.encode(numericalized_doc).cpu().numpy()
        # faiss_logger.info("The doc 304 text is: {}".format(doc_304[:200]))

        # return encoded_doc, [("304", "If you see this....")]
    
    def write_results_in_trec_format(self, results, distances, qid_query, output_path):
        faiss_id_to_doc_id = pickle.load(open("faiss_order.dump", "rb"))
        trec_string = "{qid} 0 {doc_id} {rank} {score} faiss\n"
        num_queries, num_neighbours = results.shape
        assert num_queries == len(qid_query)

        os.makedirs(output_path, exist_ok=True)
        ver_f = open("verif.log", "w")
        with open(os.path.join(output_path, "faiss.run"), "w") as f:
            for i in range(num_queries):
                faiss_ids = results[i][results[i] > -1]
                qid = qid_query[i][0]

                for j, faiss_id in enumerate(faiss_ids):
                    doc_id = faiss_id_to_doc_id[faiss_id]
                    if qid == "304":
                        in_qrels = doc_id in self.benchmark.qrels["304"]
                        # faiss_logger.info("Rank {} is {} and is in qrels: {}".format(j, doc_id, in_qrels))

                    ver_f.write("faiss {} is doc {}\n".format(faiss_id, doc_id))
                    f.write(trec_string.format(qid=qid, doc_id=doc_id, rank=j+1, score=distances[i][j]))

        ver_f.close()
        # faiss_logger.debug("The search results in TREC format are at: {}".format(output_path))

        return output_path
                
