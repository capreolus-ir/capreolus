import faiss
from collections import defaultdict
import pickle
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch
import os
import numpy as np
from capreolus import ConfigOption, constants, get_logger, Dependency, evaluator
from capreolus.sampler import CollectionSampler
from capreolus.searcher import Searcher

from . import Index
from capreolus.utils.trec import max_pool_trec_passage_run

logger = get_logger(__name__)
faiss_logger = get_logger("faiss")


@Index.register
class FAISSIndex(Index):
    module_name = "faiss"
    
    dependencies = [Dependency(key="encoder", module="encoder", name="sentencebert"), Dependency(key="index", module="index", name="anserini"), Dependency(key="benchmark", module="benchmark"), Dependency(key="searcher", module="searcher", name="BM25")] + Index.dependencies
    config_spec = [ConfigOption("isclear", False, "Whether the searcher is used with CLEAR.")]

    def create_index(self, fold=None):
        assert fold is not None

        if self.exists():
            return

        self._create_index(fold)
        donefn = self.get_index_path() / "done"
        with open(donefn, "wt") as donef:
            print("done", file=donef)

    def get_results_path(self):
        """Return an absolute path that can be used for storing results.
        The path is a function of the module's config and the configs of its dependencies.
        """

        return constants["RESULTS_BASE_PATH"] / self.get_module_path()
 
    def do_bm25_search(self):
        topics_fn = self.benchmark.topic_file
        output_dir = self.get_results_path()

        if hasattr(self.searcher, "index"):
            self.searcher.index.create_index()

        search_results_folder = self.searcher.query_from_file(topics_fn, output_dir)

        # faiss_logger.info("BM25 search results written to: %s", search_results_folder)

        return search_results_folder

    def evaluate_bm25_search(self, fold=None):
        if hasattr(self, "best_bm25_results"):
            return self.best_bm25_results

        metrics = evaluator.DEFAULT_METRICS

        best_results = evaluator.search_best_run(
            self.get_results_path(), self.benchmark, primary_metric="map", metrics=metrics, folds=fold
        )

        for fold, path in best_results["path"].items():
            logger.info("rank: fold=%s best run: %s", fold, path)

        # faiss_logger.info("rank: cross-validated results when optimizing for '%s':", "map")
        for metric, score in sorted(best_results["score"].items()):
            logger.info("%25s: %0.4f", metric, score)

        self.best_bm25_results = best_results
        
        return best_results

    def train_encoder(self, fold):
        # TODO: The BM25 search run is used to generate training data for the encoder. Remove this maybe?
        self.do_bm25_search()
        rank_results = self.evaluate_bm25_search(fold=fold)
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)
        train_run = {qid: docs for qid, docs in best_search_run.items() if qid in self.benchmark.folds[fold]["train_qids"]}
        dev_run = defaultdict(dict)
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"] and qid in self.benchmark.qrels:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= 300:
                        break
                    dev_run[qid][docid] = score

        qids = best_search_run.keys()
        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)

        self.encoder.build_model(train_run, dev_run, docids, qids)

    def get_all_docids_in_collection(self):
        from jnius import autoclass

        start = time.time()
        logger.debug("Iterating through all docids in collection")
        anserini_index = self.index
        anserini_index.create_index()
        anserini_index_path = anserini_index.get_index_path().as_posix()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        # TODO: Add check for deleted rows in the index
        all_doc_ids = [anserini_index.convert_lucene_id_to_doc_id(i) for i in range(0, anserini_index_reader.maxDoc())]
        logger.debug("That took {}".format(time.time() - start))

        return all_doc_ids

    def _create_index(self, fold):
        self.train_encoder(fold)
        sub_index = faiss.IndexFlatIP(self.encoder.hidden_size)
        faiss_index = faiss.IndexIDMap2(sub_index)

        collection_docids = self.get_all_docids_in_collection()

        logger.debug("Preprocessing the extractor")
        self.encoder.extractor.preprocess([], collection_docids, topics=self.benchmark.topics[self.benchmark.query_type])
        logger.debug("Preparing the collection sampler")
        dataset = CollectionSampler()
        dataset.prepare(
            collection_docids, None, self.encoder.extractor, relevance_level=self.benchmark.relevance_level
        )

        BATCH_SIZE = 64
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.model.to(device)
        self.encoder.model.eval()

        doc_id_to_faiss_id = {}
        for bi, batch in tqdm(enumerate(dataloader), desc="FAISS index creation"):
            batch = {k: v.to(device) if not isinstance(v, list) else v for k, v in batch.items()}
            doc_ids = batch["posdocid"]
            faiss_ids_for_batch = []

            for i, doc_id in enumerate(doc_ids):
                generated_faiss_id = bi * BATCH_SIZE + i
                doc_id_to_faiss_id[doc_id] = generated_faiss_id
                faiss_ids_for_batch.append(generated_faiss_id)

            with torch.no_grad():
                doc_emb = self.encoder.encode_doc(batch["posdoc"], batch["posdoc_mask"]).cpu().numpy()
            # self.doc_embs.append((batch["posdocid"], doc_emb))
            # assert doc_emb.shape == (BATCH_SIZE, self.encoder.hidden_size)

            faiss_ids_for_batch = np.array(faiss_ids_for_batch, dtype=np.long).reshape(-1, )
            faiss_index.add_with_ids(doc_emb, faiss_ids_for_batch)

        os.makedirs(self.get_cache_path(), exist_ok=True)
        doc_id_to_faiss_id_fn = os.path.join(self.get_cache_path(), "doc_id_to_faiss_id.dump")
        pickle.dump(doc_id_to_faiss_id, open(doc_id_to_faiss_id_fn, "wb"), protocol=-1)

        faiss_id_to_doc_id_fn = os.path.join(self.get_cache_path(), "faiss_id_to_doc_id.dump")
        faiss_id_to_doc_id = {faiss_id: doc_id for doc_id, faiss_id in doc_id_to_faiss_id.items()}
        pickle.dump(faiss_id_to_doc_id, open(faiss_id_to_doc_id_fn, "wb"), protocol=-1)

        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(self.get_index_path(), "faiss.index"))

        # TODO: write the "done" file

    def reweight_using_bm25_scores(self, distances, results, qid_query, fold, lambda_test=0.5):
        """
        Takes the cosine similarity scores that FAISS produces and re-weights them as specified in the CLEAR paper
        """
        faiss_id_to_doc_id_fn = os.path.join(self.get_cache_path(), "faiss_id_to_doc_id.dump")
        rank_results = self.evaluate_bm25_search(fold=fold)
        best_search_run_path = rank_results["path"][fold]
        bm25_run = Searcher.load_trec_run(best_search_run_path)
        faiss_id_to_doc_id = pickle.load(open(faiss_id_to_doc_id_fn, "rb"))
        num_queries, num_neighbours = results.shape

        for i in range(num_queries):
            qid = qid_query[i][0]
            for j, faiss_id in enumerate(results[i]):
                if faiss_id == -1:
                    continue
                doc_id = faiss_id_to_doc_id[faiss_id]
                distances[i][j] = lambda_test * bm25_run[qid].get(doc_id, 0) + distances[i, j]

        return distances

    def faiss_search(self, topic_vectors, k, qid_query, fold):
        logger.debug("starting faiss search")
        faiss_index = faiss.read_index(os.path.join(self.get_index_path(), "faiss.index"))
        logger.debug("Faiss search done")

        distances, results = faiss_index.search(topic_vectors, k)
        if self.config["isclear"]:
            faiss_logger.info("Reweighting FAISS scores for CLEAR")
            distances = self.reweight_using_bm25_scores(distances, results, qid_query, fold)

        return distances, results

    def manual_search_train_set(self, topic_vectors, qid_query, fold):
        # Get the dev_run BM25 results for this fold.
        rank_results = self.evaluate_bm25_search(fold=fold)
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        bm25_train_run = {qid: docs_to_score for qid, docs_to_score in best_search_run.items() if
                    qid in self.benchmark.folds[fold]["train_qids"]}

        metrics = self.manual_search(topic_vectors, qid_query, bm25_train_run)

        faiss_logger.info("Train Fold %s manual metrics: %s", fold,
                          " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def manual_search_dev_set(self, topic_vectors, qid_query, fold, tag):
        # Get the dev_run BM25 results for this fold.
        rank_results = self.evaluate_bm25_search(fold=fold)
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        bm25_dev_run = {qid: docs_to_score for qid, docs_to_score in best_search_run.items() if
                    qid in self.benchmark.folds[fold]["predict"]["dev"]}

        metrics = self.manual_search(topic_vectors, qid_query, bm25_dev_run)

        faiss_logger.info("%s: Dev Fold %s manual metrics: %s", tag, fold,
                          " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def manual_search_test_set(self, topic_vectors, qid_query, fold, tag):
        # Get the dev_run BM25 results for this fold.
        rank_results = self.evaluate_bm25_search(fold=fold)
        best_search_run_path = rank_results["path"][fold]
        best_search_run = Searcher.load_trec_run(best_search_run_path)

        bm25_dev_run = {qid: docs_to_score for qid, docs_to_score in best_search_run.items() if
                    qid in self.benchmark.folds[fold]["predict"]["test"]}

        metrics = self.manual_search(topic_vectors, qid_query, bm25_dev_run)

        faiss_logger.info("%s: Test Fold %s manual metrics: %s", tag, fold,
                          " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

    def manual_search(self, topic_vectors, qid_query, bm25_run):
        """
        Manual search is different from the normal FAISS search.
        For a given qid, we manually calculate cosine scores for only those documents retrieved by BM25 for this qid.
        This helps in debugging - the scores should be the same as the final validation scored obtained while training the encoder
        A "real" FAISS search would calculate the cosine score by comparing a qid with _every_ other document in the index - not just the docs retrieved for the query by BM25
        """
        # Load some dicts that will allow us to map faiss index ids to docids
        faiss_index = faiss.read_index(os.path.join(self.get_index_path(), "faiss.index"))
        doc_id_to_faiss_id_fn = os.path.join(self.get_cache_path(), "doc_id_to_faiss_id.dump")
        doc_id_to_faiss_id = pickle.load(open(doc_id_to_faiss_id_fn, "rb"))

        results = defaultdict(list)
        run = {}
        faiss_logger.debug("Starting manual search")

        for i, (qid, query) in tqdm(enumerate(qid_query), desc="Manual search"):
            if qid not in bm25_run:
                continue

            query_emb = topic_vectors[i].reshape(768)
            for doc_id in bm25_run[qid].keys():
                faiss_id = doc_id_to_faiss_id[doc_id]
                doc_emb = faiss_index.reconstruct(faiss_id).reshape(768)
                score = np.dot(query_emb, doc_emb)
                score = score.astype(np.float16).item()
                results[qid].append((score, doc_id))

        for qid in results:
            score_doc_id = results[qid]
            for i, (score, doc_id) in enumerate(sorted(score_doc_id, reverse=True)):
                run.setdefault(qid, {})[doc_id] = score

        if self.benchmark.module_name == "robust04passages":
            run = max_pool_trec_passage_run(run)

        metrics = evaluator.eval_runs(run, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)

        return metrics

    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

