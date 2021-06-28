import torch
from collections import defaultdict
import faiss
import random
import re
import pickle
from tqdm import tqdm
import subprocess
import os
import numpy as np
from capreolus import ConfigOption, Dependency, constants, evaluator, Anserini
from capreolus.utils.trec import load_trec_topics, pool_trec_passage_run
from capreolus import get_logger

from . import Searcher
import capreolus.evaluator

logger = get_logger(__name__)
faiss_logger = get_logger("faiss")


@Searcher.register
class FAISSSearcher(Searcher):
    """
    Use FAISS to do approximate nearest-neighbor search on embeddings created using sentenceBERT
    """

    module_name = "faiss"
    config_spec = [
        ConfigOption("field", "title", "The query field that should be used for retrieval")
    ]

    dependencies = [Dependency(key="index", module="index", name="faiss"), Dependency(key="benchmark", module="benchmark")]

    def query_from_file(self, topicsfn, output_path, fold=None):
        raise Exception("Call _query_from_file directly - you need to pass an encoder instance")

    def do_search(self, topic_vectors, qid_query, numshards, docs_per_shard, fold, output_path, filename, tag):
        # A manual search is done over the docs in dev_run - this way for each qid, we only score the docids that BM25 retrieved for it
        search_results_folder = os.path.join(output_path, "rank")
        best_bm25_results = evaluator.search_best_run(
            search_results_folder, self.benchmark, primary_metric="map", metrics=evaluator.DEFAULT_METRICS, folds=fold
        )
        best_search_run_path = best_bm25_results["path"][fold]
        bm25_run = Searcher.load_trec_run(best_search_run_path)

        docs_in_bm25_run = []
        for qid, docid_to_score in bm25_run.items():
            docs_in_bm25_run.extend(list(docid_to_score.keys()))
        docs_in_bm25_run = set(docs_in_bm25_run)

        # self.index.manual_search_train_set(topic_vectors, qid_query, fold)
        distances, results = self.index.faiss_search(topic_vectors, 1000, docs_in_bm25_run, numshards, docs_per_shard, fold, output_path)
        # self.calc_faiss_search_metrics_for_train_set(distances, results, qid_query, fold)
        self.calc_faiss_search_metrics_for_dev_set(distances, results, qid_query, fold, tag, output_path)
        self.calc_faiss_search_metrics_for_test_set(distances, results, qid_query, fold, tag, output_path)
        distances = distances.astype(np.float16)
        self.write_results_in_trec_format(results, distances, qid_query, output_path, fold, filename=filename)

        # Don't calculate re-rank metrics for PRF
        if not "topdoc" in tag:
            self.index.manual_search_dev_set(bm25_run, topic_vectors, qid_query, fold, docs_per_shard, output_path, tag)
            self.index.manual_search_test_set(bm25_run, topic_vectors, qid_query, fold, docs_per_shard, output_path, tag)

        return distances, results

    def _query_from_file(self, encoder, topics, output_path, numshards, docs_per_shard, fold=None):
        assert fold is not None

        # `qid_query` contains (qid, query) tuples in the order they were encoded
        topic_vectors, qid_query = self.create_topic_vectors(encoder, topics, fold, topicfield=self.config["field"])
        normal_distances, normal_results = self.do_search(topic_vectors, qid_query, numshards, docs_per_shard, fold, output_path, "faiss_{}.run".format(fold), "normal")
        # self.interpolate(self.index.get_results_path(), os.path.join(output_path, "faiss_{}.run".format(fold)), fold, "normal")

        # rm3_expanded_topics = self.rm3_expand_queries(os.path.join(output_path, "faiss.run"), topicfield="title")
        # rm3_expanded_topic_vectors, rm3_qid_query = self.create_topic_vectors(rm3_expanded_topics, fold, topicfield="title")
        # self.do_search(rm3_expanded_topic_vectors, rm3_qid_query, fold, output_path, "faiss_rm3_expanded.run", "rm3")

        topdoc_expanded_topic_vectors, topdoc_qid_query = self.topdoc_expand_queries(qid_query, topic_vectors, normal_results, fold, output_path, docs_per_shard, k=1)
        self.do_search(topdoc_expanded_topic_vectors, topdoc_qid_query, numshards, docs_per_shard, fold, output_path, "faiss_topdoc_expanded_{}_top1.run".format(fold), "topdoc-1:")
        topdoc_expanded_topic_vectors, topdoc_qid_query = self.topdoc_expand_queries(qid_query, topic_vectors, normal_results, fold, output_path, docs_per_shard, k=3)
        self.do_search(topdoc_expanded_topic_vectors, topdoc_qid_query, numshards, docs_per_shard, fold, output_path, "faiss_topdoc_expanded_{}_top3.run".format(fold), "topdoc-3:")
        topdoc_expanded_topic_vectors, topdoc_qid_query = self.topdoc_expand_queries(qid_query, topic_vectors, normal_results, fold, output_path, docs_per_shard, k=5)
        self.do_search(topdoc_expanded_topic_vectors, topdoc_qid_query, numshards, docs_per_shard, fold, output_path, "faiss_topdoc_expanded_{}_top5.run".format(fold), "topdoc-5:")
        # topdoc_expanded_topic_vectors, topdoc_qid_query = self.topdoc_expand_queries(qid_query, topic_vectors, normal_results, fold, k=7)
        # self.do_search(topdoc_expanded_topic_vectors, topdoc_qid_query, fold, output_path, "faiss_topdoc_expanded_{}.run".format(fold), "topdoc-7:")
        #
        # self.interpolate(self.index.get_results_path(), os.path.join(output_path, "faiss_topdoc_expanded.run"), fold, "topdoc")
        # # Deleting the results obtained using the expanded queries
        # # os.remove(os.path.join(output_path, "faiss_rm3_expanded.run"))
        # os.remove(os.path.join(output_path, "faiss_topdoc_expanded_{}.run".format(fold)))

        return output_path

    def build_encoder(self, fold):
        """
        TODO: Deprecate this method. We should not rely on BM25 search results to load the pre-trained encoder.
        Solution: Do not use BM25 run to form the TrainTripletSampler while training the encoder
        """

        rank_results = self.index.evaluate_bm25_search(fold)
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)
        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}

        dev_run = defaultdict(dict)
        # Limit validation to top 100 BM25 results
        # This is possible because in python 3.6+, dictionaries preserve insertion order
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"] and qid in self.benchmark.qrels:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= 100:
                        break
                    dev_run[qid][docid] = score

        encoder_qids = best_search_run.keys()
        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)

        self.index.encoder.build_model(train_run, dev_run, docids, encoder_qids)

    def create_topic_vectors(self, encoder, topics, fold, topicfield="title"):
        """
        Creates a tensor of shape (num_queries, emb_size). Uses all the topics available in the dataset. Filtering based on folds is done later
        """
        if topicfield == "combined":
            qid_query = []
            for qid in topics:
                query_title = topics["title"][qid]
                query_desc = topics["desc"][qid]
                qid_query.append((qid, "{}. {}".format(query_title, query_desc)))

            qid_query = sorted(qid_query)
        else:
            qid_query = sorted([(qid, query) for qid, query in topics[topicfield].items() if qid in self.benchmark.folds[fold]["predict"]["dev"] or qid in self.benchmark.folds[fold]["predict"]["test"]])

        tokenizer = encoder.extractor.tokenizer
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
                topic_vector = encoder.encode_query(numericalized_query, mask).cpu().numpy()
                topic_vectors.append(topic_vector)
                
        return np.concatenate(topic_vectors, axis=0), qid_query

    def topdoc_expand_queries(self, qid_query, original_topic_vectors, results, fold, output_path, docs_per_shard, k=1):
        topic_vectors = []

        for i, (qid, query) in enumerate(qid_query):
            # topdocs are stored according to increasing cosine similarity. So the top-doc is at the very end
            # reverse the array so that the best doc is the first doc
            topdocs = results[i][results[i] > -1][:k]

            averaged_topdoc_emb = [original_topic_vectors[i]]
            for j in range(1, k+1):
                topdoc_id = int(topdocs[-j])
                shard_id = topdoc_id // docs_per_shard
                shard = faiss.read_index(os.path.join(output_path, "shard_{}_faiss_{}.index".format(shard_id, fold)))

                topdoc_emb = shard.reconstruct(topdoc_id)
                averaged_topdoc_emb.append(topdoc_emb)

            averaged_topdoc_emb = np.array(averaged_topdoc_emb)
            averaged_topdoc_emb = np.mean(averaged_topdoc_emb, axis=0)

            topic_vectors.append(averaged_topdoc_emb)

        topic_vectors = np.array(topic_vectors)

        return topic_vectors, qid_query

    def rm3_expand_queries(self, faiss_run_file, topicfield="title"):
        index_path = self.index.index.get_index_path()
        topicsfn = self.benchmark.get_topics_file()
        os.makedirs(self.get_cache_path(), exist_ok=True)
        output_path = os.path.join(self.get_cache_path(), "expanded_queries.txt")

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

        expanded_topics = self.load_expanded_topics(output_path, topicfield)

        return expanded_topics

    def load_expanded_topics(self, expanded_topics_fn, topicfield):
        expanded_topics = {topicfield: {}}
        with open(expanded_topics_fn, "r") as f:
            for line in f:
                qid, boosted_query = line.split("\t")
                query_terms = re.findall('\(.*?\)', boosted_query)
                query_terms = [s[1:-1] for s in query_terms]
                expanded_topics[topicfield][qid] = " ".join(query_terms)
                if random.random() > 0.9:
                    logger.debug("Expanded query {} is: {}".format(qid, " ".join(query_terms)))

        return expanded_topics

    def write_results_in_trec_format(self, results, distances, qid_query, output_path, fold, filename="faiss.run"):
        faiss_id_to_doc_id_fn = os.path.join(output_path, "faiss_id_to_doc_id_{}.dump".format(fold))
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

    def calc_faiss_search_metrics_for_dev_set(self, distances, results, qid_query, fold, tag, output_path):
        valid_qids = [qid for qid in self.benchmark.folds[fold]["predict"]["dev"]]
        metrics = self.calc_faiss_search_metrics(distances, results, qid_query, valid_qids, fold, output_path)
        faiss_logger.info("%s: FAISS dev set metrics: %s", tag, " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def calc_faiss_search_metrics_for_test_set(self, distances, results, qid_query, fold, tag, output_path):
        valid_qids = [qid for qid in self.benchmark.folds[fold]["predict"]["test"]]
        metrics = self.calc_faiss_search_metrics(distances, results, qid_query, valid_qids, fold, output_path)
        faiss_logger.info("%s: FAISS test set metrics: %s", tag,
                          " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def calc_faiss_search_metrics(self, distances, results, qid_query, valid_qids, fold, output_path):
        faiss_id_to_doc_id_fn = os.path.join(output_path, "faiss_id_to_doc_id_{}.dump".format(fold))
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

        if hasattr(self.benchmark, "need_pooling") and self.benchmark.need_pooling:
             run = pool_trec_passage_run(run, strategy=self.benchmark.config["pool"])

        metrics = evaluator.eval_runs(run, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)

        return metrics

    def interpolate(self, bm25_run_folder, faiss_run_fn, fold, tag):
        best_run = evaluator.search_best_run(bm25_run_folder, self.benchmark, "map", metrics=evaluator.DEFAULT_METRICS, folds=fold)
        bm25_run = Searcher.load_trec_run(best_run["path"][fold])
        faiss_run = Searcher.load_trec_run(faiss_run_fn)
        qids = self.benchmark.folds[fold]["predict"]["test"]

        faiss_favor = 0
        bm25_favor = 0

        for qid in qids:
            if qid not in self.benchmark.qrels:
                continue

            qrels = self.benchmark.qrels[qid]
            bm25_retrieved = set([docid for docid in bm25_run[qid].keys() if docid in qrels and qrels[docid] >= 1])
            faiss_retrieved = set([docid for docid in faiss_run[qid].keys() if docid in qrels and qrels[docid] >= 1])
            faiss_favor += len(faiss_retrieved - bm25_retrieved)
            bm25_favor += len(bm25_retrieved - faiss_retrieved)

        faiss_logger.info("{}: FAISS retrieved {} reldocs that BM25 did not".format(tag, faiss_favor))
        faiss_logger.info("{}: BM25 retrieved {} reldocs that FAISS did not".format(tag, bm25_favor))

        bm25_max = 0
        bm25_min = np.inf
        for qid, docid_to_score in bm25_run.items():
            max_, min_ = max(docid_to_score.values()), min(docid_to_score.values())
            if max_ > bm25_max:
                bm25_max = max_
            if min_ < bm25_min:
                bm25_min = min_

        faiss_max = 0
        faiss_min = np.inf
        for qid, docid_to_score in faiss_run.items():
            max_, min_ = max(docid_to_score.values()), min(docid_to_score.values())
            if max_ > faiss_max:
                faiss_max = max_
            if min_ < faiss_min:
                faiss_min = min_

        interpolated_run = {}
        # Interpolate the scores
        for qid in qids:
            if qid not in self.benchmark.qrels:
                continue

            docids = set(bm25_run[qid].keys()).union(set(faiss_run[qid].keys()))
            for docid in docids:
                if docid in bm25_run[qid] and docid in faiss_run[qid]:
                    normalized_bm25 = (bm25_run[qid][docid] - bm25_min) / (bm25_max - bm25_min)
                    normalized_faiss = (faiss_run[qid][docid] - faiss_min) / (faiss_max - faiss_min)
                    interpolated_run.setdefault(qid, {})[docid] = (normalized_faiss + normalized_bm25) / 2
                elif docid in bm25_run[qid] and docid not in faiss_run[qid]:
                    interpolated_run.setdefault(qid, {})[docid] = (bm25_run[qid][docid] - bm25_min) / (bm25_max - bm25_min)
                elif docid not in bm25_run[qid] and docid in faiss_run[qid]:
                    interpolated_run.setdefault(qid, {})[docid] = (faiss_run[qid][docid] - faiss_min) / (faiss_max - faiss_min)

        if hasattr(self.benchmark, "need_pooling") and self.benchmark.need_pooling:
            interpolated_run = pool_trec_passage_run(interpolated_run, strategy=self.benchmark.config["pool"])

        metrics = evaluator.eval_runs(interpolated_run, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)
        faiss_logger.info("%s: Interpolated Test Fold %s metrics: %s", tag, fold, " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
