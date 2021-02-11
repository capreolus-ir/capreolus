import torch
import random
import re
import pickle
from tqdm import tqdm
import subprocess
import os
import numpy as np
from capreolus import ConfigOption, Dependency, constants, evaluator, Anserini
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

        self.build_encoder(fold)
        topics = load_trec_topics(topicsfn)
        # `qid_query` contains (qid, query) tuples in the order they were encoded
        topic_vectors, qid_query = self.create_topic_vectors(topics, fold)
        # A manual search is done over the docs in dev_run - this way for each qid, we only score the docids that BM25 retrieved for it
        # self.index.manual_search_train_set(topic_vectors, qid_query, fold)
        self.index.manual_search_dev_set(topic_vectors, qid_query, fold)
        self.index.manual_search_test_set(topic_vectors, qid_query, fold)
        distances, results = self.index.faiss_search(topic_vectors, 1000, qid_query, fold)
        # self.calc_faiss_search_metrics_for_train_set(distances, results, qid_query, fold)
        self.calc_faiss_search_metrics_for_dev_set(distances, results, qid_query, fold)
        self.calc_faiss_search_metrics_for_test_set(distances, results, qid_query, fold)
        distances = distances.astype(np.float16)
        self.write_results_in_trec_format(results, distances, qid_query, output_path, filename="faiss.run")

        expanded_topics = self.expand_queries(os.path.join(output_path, "faiss.run"))
        expanded_topic_vectors, qid_query = self.create_topic_vectors(expanded_topics)
        self.index.manual_search_dev_set(expanded_topic_vectors, qid_query, fold)
        self.index.manual_search_test_set(expanded_topic_vectors, qid_query, fold)
        distances, results = self.index.faiss_search(expanded_topic_vectors, 1000, qid_query, fold)
        self.calc_faiss_search_metrics_for_dev_set(distances, results, qid_query, fold)
        self.calc_faiss_search_metrics_for_test_set(distances, results, qid_query, fold)
        distances = distances.astype(np.float16)
        self.write_results_in_trec_format(results, distances, qid_query, output_path, filename="faiss_expanded.run")

        # Deleting the results obtained using the unexpanded queries
        os.remove(os.path.join(output_path, "faiss.run"))

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

    def create_topic_vectors(self, topics, fold):
        """
        Creates a tensor of shape (num_queries, emb_size). Uses all the topics available in the dataset. Filtering based on folds is done later
        """

        # TODO: Use the test qids in the below line

        qid_query = sorted([(qid, query) for qid, query in topics["title"].items() if qid in self.benchmark.folds[fold]["predict"]["dev"] or qid in self.benchmark.folds[fold]["predict"]["test"]])
        tokenizer = self.index.encoder.extractor.tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        topic_vectors = []
        with torch.no_grad():
            for qid, query in tqdm(qid_query, desc="Encode topics"):
                query_toks = tokenizer.tokenize(query)[:510]
                numericalized_query = tokenizer.convert_tokens_to_ids(["[CLS]"] + query_toks + ["[SEP]"])
                mask = torch.tensor([1 if t != 0 else 0 for t in numericalized_query], dtype=torch.long)
                mask = mask.to(device)
                mask = mask.reshape(1, -1)
                numericalized_query = torch.tensor(numericalized_query).to(device)
                numericalized_query = numericalized_query.reshape(1, -1)
                topic_vector = self.index.encoder.encode_query(numericalized_query, mask).cpu().numpy()
                topic_vectors.append(topic_vector)
                
        return np.concatenate(topic_vectors, axis=0), qid_query

    def expand_queries(self, faiss_run_file):
        index_path = self.index.index.get_index_path()
        topicsfn = self.benchmark.topic_file
        output_path = os.path.join(self.get_cache_path(), "expanded_queries.txt")
        topicfield = "title"

        cmd = [
            "java",
            "-classpath",
            "/home/kjose/anserini/target/anserini-0.10.2-SNAPSHOT-fatjar.jar",
            "-Xms512M",
            "-Xmx31G",
            "-Dapp.name=ExpandQueries",
            "io.anserini.search.ExpandQueries",
            "-topicreader",
            "Trec",
            "-runFile",
            faiss_run_file,
            "-index",
            index_path,
            "-topics",
            topicsfn,
            "-output",
            output_path,
            "-topicfield",
            topicfield,
            "-stemmer",
            "none",
        ]
        app = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for line in app.stdout:
            Anserini.filter_and_log_anserini_output(line, logger)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

        logger.info("Expanded queries written to: {}".format(output_path))

        expanded_topics = self.load_expanded_topics(output_path)

        return expanded_topics

    def load_expanded_topics(self, expanded_topics_fn):
        expanded_topics = {"title": {}}
        with open(expanded_topics_fn, "r") as f:
            for line in f:
                qid, boosted_query = line.split("\t")
                query_terms = re.findall('\(.*?\)', boosted_query)
                query_terms = [s[1:-1] for s in query_terms]
                expanded_topics["title"][qid] = " ".join(query_terms)
                if random.random() > 0.9:
                    logger.debug("Expanded query {} is: {}".format(qid, " ".join(query_terms)))

        return expanded_topics

    def write_results_in_trec_format(self, results, distances, qid_query, output_path, filename="faiss.run"):
        faiss_id_to_doc_id_fn = os.path.join(self.index.get_cache_path(), "faiss_id_to_doc_id.dump")
        faiss_id_to_doc_id = pickle.load(open(faiss_id_to_doc_id_fn, "rb"))
        trec_string = "{qid} 0 {doc_id} {rank} {score} faiss\n"
        num_queries, num_neighbours = results.shape
        assert num_queries == len(qid_query)

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, filename), "w") as f:
            for i in range(num_queries):
                faiss_ids = results[i][results[i] > -1]
                qid = qid_query[i][0]
                for j, faiss_id in enumerate(faiss_ids):
                    doc_id = faiss_id_to_doc_id[faiss_id]
                    f.write(trec_string.format(qid=qid, doc_id=doc_id, rank=j+1, score=distances[i][j]))

        return output_path

    def calc_faiss_search_metrics_for_train_set(self, distances, results, qid_query, fold):
        valid_qids = [qid for qid in self.benchmark.folds[fold]["train_qids"]]
        metrics = self.calc_faiss_search_metrics(distances, results, qid_query, valid_qids)
        faiss_logger.info("FAISS train set metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def calc_faiss_search_metrics_for_dev_set(self, distances, results, qid_query, fold):
        valid_qids = [qid for qid in self.benchmark.folds[fold]["predict"]["dev"]]
        metrics = self.calc_faiss_search_metrics(distances, results, qid_query, valid_qids)
        faiss_logger.info("FAISS dev set metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def calc_faiss_search_metrics_for_test_set(self, distances, results, qid_query, fold):
        valid_qids = [qid for qid in self.benchmark.folds[fold]["predict"]["test"]]
        metrics = self.calc_faiss_search_metrics(distances, results, qid_query, valid_qids)
        faiss_logger.info("FAISS test set metrics: %s",
                          " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def calc_faiss_search_metrics(self, distances, results, qid_query, valid_qids):
        faiss_id_to_doc_id_fn = os.path.join(self.index.get_cache_path(), "faiss_id_to_doc_id.dump")
        faiss_id_to_doc_id = pickle.load(open(faiss_id_to_doc_id_fn, "rb"))
        num_queries, num_neighbours = results.shape
        run = {}

        for i in range(num_queries):
            qid = qid_query[i][0]
            if qid not in valid_qids:
                continue

            faiss_ids = results[i][results[i] > -1]
            for j, faiss_id in enumerate(faiss_ids):
                doc_id = faiss_id_to_doc_id[faiss_id]
                run.setdefault(qid, {})[doc_id] = distances[i][j].item()

        metrics = evaluator.eval_runs(run, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)

        return metrics
