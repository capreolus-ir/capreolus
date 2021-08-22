import faiss
from collections import defaultdict
import pickle
import time
from tqdm import tqdm
import torch
import os
import numpy as np
from capreolus import ConfigOption, constants, get_logger, Dependency, evaluator
from capreolus.sampler import CollectionSampler
from capreolus.searcher import Searcher

from . import Index
from capreolus.utils.trec import pool_trec_passage_run

logger = get_logger(__name__)
faiss_logger = get_logger("faiss")


@Index.register
class FAISSIndex(Index):
    module_name = "faiss"

    dependencies = [
        Dependency(key="index", module="index", name="anserini"),
        Dependency(key="benchmark", module="benchmark"),
    ] + Index.dependencies
    config_spec = [ConfigOption("isclear", False, "Whether the searcher is used with CLEAR encoder.")]

    def exists(self, fold=None):
        return False

    def create_index(self, fold=None):
        logger.error("FAISSIndex does not implement create_index()")
        pass

    def get_results_path(self):
        """Return an absolute path that can be used for storing results.
        The path is a function of the module's config and the configs of its dependencies.
        """

        return constants["RESULTS_BASE_PATH"] / self.get_module_path()

    def create_shard(self, encoder, shard_id, offset, doc_ids, fold, output_dir):
        """
        Creates a FAISS index that contains the vectors for the supplied docids
        :params:
        shard_id - an integer indicating the id of the shard
        offset - shard_id * num_docs_per_shard. This is used to generate unique ids for docs
        """
        faiss_logger.info("Creating shard: {} with {} docs".format(shard_id, len(doc_ids)))
        sub_index = faiss.IndexFlatIP(encoder.hidden_size)
        faiss_index = faiss.IndexIDMap2(sub_index)
        encoder.extractor.preprocess([], doc_ids, topics=self.benchmark.topics[self.benchmark.query_type])

        dataset = CollectionSampler()
        dataset.prepare(doc_ids, None, encoder.extractor, relevance_level=self.benchmark.relevance_level)

        BATCH_SIZE = 64
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.model.to(device)
        encoder.model.eval()

        doc_id_to_faiss_id = {}
        for bi, batch in tqdm(enumerate(dataloader), desc="FAISS index creation"):
            batch = {k: v.to(device) if not isinstance(v, list) else v for k, v in batch.items()}
            doc_ids = batch["posdocid"]
            faiss_ids_for_batch = []

            for i, doc_id in enumerate(doc_ids):
                generated_faiss_id = bi * BATCH_SIZE + i + offset
                doc_id_to_faiss_id[doc_id] = generated_faiss_id
                faiss_ids_for_batch.append(generated_faiss_id)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    doc_emb = encoder.encode_doc(batch["posdoc"], batch["posdoc_mask"]).cpu().numpy()

            faiss_ids_for_batch = np.array(faiss_ids_for_batch, dtype=np.long).reshape(
                -1,
            )
            faiss_index.add_with_ids(doc_emb, faiss_ids_for_batch)

        os.makedirs(self.get_cache_path(), exist_ok=True)
        doc_id_to_faiss_id_fn = os.path.join(output_dir, "shard_{}_doc_id_to_faiss_id_{}.dump".format(shard_id, fold))
        pickle.dump(doc_id_to_faiss_id, open(doc_id_to_faiss_id_fn, "wb"), protocol=-1)

        faiss_id_to_doc_id_fn = os.path.join(output_dir, "shard_{}_faiss_id_to_doc_id_{}.dump".format(shard_id, fold))
        faiss_id_to_doc_id = {faiss_id: doc_id for doc_id, faiss_id in doc_id_to_faiss_id.items()}
        pickle.dump(faiss_id_to_doc_id, open(faiss_id_to_doc_id_fn, "wb"), protocol=-1)

        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(output_dir, "shard_{}_faiss_{}.index".format(shard_id, fold)))
        faiss_logger.info("Shard: {} written to disk".format(shard_id))

    def reweight_using_bm25_scores(self, distances, results, qid_query, fold, lambda_test=0.5):
        """
        Takes the cosine similarity scores that FAISS produces and re-weights them as specified in the CLEAR paper.
        This method is not used right now. Use `self.config["isclear"]` to call this method based on user-specified
        config
        """
        faiss_id_to_doc_id_fn = os.path.join(self.get_cache_path(), "faiss_id_to_doc_id_{}.dump".format(fold))
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

    def faiss_search(self, topic_vectors, k, docs_in_bm25_run, numshards, docs_per_shard, fold, output_path):
        start_time = time.time()
        aggregated_faiss_id_to_doc_id = {}
        aggregated_doc_id_to_faiss_id = {}
        count_map = defaultdict(lambda: 0)
        aggregated_distances, aggregated_ids = np.zeros((len(topic_vectors), numshards * k)), np.zeros(
            (len(topic_vectors), numshards * k)
        )
        bm25_sub_index = faiss.IndexFlatIP(768)
        bm25_faiss_index = faiss.IndexIDMap2(bm25_sub_index)

        for shard_id in range(numshards):
            offset = shard_id * docs_per_shard
            filename = os.path.join(output_path, "shard_{}_faiss_{}.index".format(shard_id, fold))
            assert os.path.isfile(filename), "shard {} not found".format(filename)
            faiss_shard = faiss.read_index(os.path.join(output_path, filename))
            distances, results = faiss_shard.search(topic_vectors, k)
            aggregated_distances[:, shard_id * k : (shard_id + 1) * k] = distances
            aggregated_ids[:, shard_id * k : (shard_id + 1) * k] = results

            faiss_id_to_doc_id = pickle.load(
                open(os.path.join(output_path, "shard_{}_faiss_id_to_doc_id_{}.dump".format(shard_id, fold)), "rb")
            )
            doc_id_to_faiss_id = pickle.load(
                open(os.path.join(output_path, "shard_{}_doc_id_to_faiss_id_{}.dump".format(shard_id, fold)), "rb")
            )

            for faiss_id, doc_id in faiss_id_to_doc_id.items():
                if doc_id in docs_in_bm25_run:
                    doc_emb = faiss_shard.reconstruct(faiss_id).reshape(1, 768)
                    bm25_faiss_index.add_with_ids(doc_emb, np.array([faiss_id]))

                assert (
                    faiss_id >= offset
                ), "faiss_id {} for shard: {} not greater than the offset: {} (docs_per_shard is: {})".format(
                    faiss_id, shard_id, offset, docs_per_shard
                )
                # assert faiss_id <= (shard_id + 1) * docs_per_shard, "faiss_id {} for shard: {} is greater than the next offset: {} (docs_per_shard is: {})".format(faiss_id, shard_id, (shard_id + 1) * docs_per_shard, docs_per_shard)
                count_map[faiss_id] += 1

            aggregated_faiss_id_to_doc_id.update(faiss_id_to_doc_id)
            aggregated_doc_id_to_faiss_id.update(doc_id_to_faiss_id)

        faiss.write_index(bm25_faiss_index, os.path.join(output_path, "bm25_faiss_{}.index".format(fold)))

        # result_heap.finalize()
        pickle.dump(
            aggregated_faiss_id_to_doc_id,
            open(os.path.join(output_path, "faiss_id_to_doc_id_{}.dump".format(fold)), "wb"),
            protocol=-1,
        )
        pickle.dump(
            aggregated_doc_id_to_faiss_id,
            open(os.path.join(output_path, "doc_id_to_faiss_id_{}.dump".format(fold)), "wb"),
            protocol=-1,
        )

        indices = np.argsort(aggregated_distances, axis=1)
        # Cosine similarity. Higher is better
        aggregated_distances = np.take_along_axis(aggregated_distances, indices, 1)
        aggregated_distances = aggregated_distances[:, (numshards - 1) * k : numshards * k]
        aggregated_ids = np.take_along_axis(aggregated_ids, indices, 1)
        aggregated_ids = aggregated_ids[:, (numshards - 1) * k : numshards * k]

        assert aggregated_distances.shape == (len(topic_vectors), k)
        assert aggregated_ids.shape == (len(topic_vectors), k)

        temp = 0
        for faiss_id, count in count_map.items():
            if count > 1:
                logger.info("{} has count {}".format(faiss_id, count))
                temp += 1
        logger.info("temp is {}".format(temp))
        assert temp == 0

        faiss_logger.info("Faiss search took {}".format(time.time() - start_time))
        return aggregated_distances, aggregated_ids

    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        return self.index.get_doc(docid)

    def manual_search_dev_set(self, bm25_run, topic_vectors, qid_query, fold, docs_per_shard, output_path, tag):
        # Get the dev_run BM25 results for this fold.
        bm25_dev_run = {
            qid: docs_to_score for qid, docs_to_score in bm25_run.items() if qid in self.benchmark.folds[fold]["predict"]["dev"]
        }

        metrics = self.manual_search(topic_vectors, qid_query, bm25_dev_run, fold, docs_per_shard, output_path)

        faiss_logger.info(
            "%s: Dev Fold %s manual metrics: %s",
            tag,
            fold,
            " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]),
        )

    def manual_search_test_set(self, bm25_run, topic_vectors, qid_query, fold, docs_per_shard, output_path, tag):
        bm25_test_run = {
            qid: docs_to_score for qid, docs_to_score in bm25_run.items() if qid in self.benchmark.folds[fold]["predict"]["test"]
        }

        metrics = self.manual_search(topic_vectors, qid_query, bm25_test_run, fold, docs_per_shard, output_path)

        faiss_logger.info(
            "%s: Test Fold %s manual metrics: %s",
            tag,
            fold,
            " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]),
        )

    def manual_search(self, topic_vectors, qid_query, bm25_run, fold, docs_per_shard, output_path):
        """
        Manual search is different from the normal FAISS search.
        For a given qid, we manually calculate cosine scores for only those documents retrieved by BM25 for this qid.
        This helps in debugging - the scores should be the same as the final validation scored obtained while training the encoder
        A "real" FAISS search would calculate the cosine score by comparing a qid with _every_ other document in the index - not just the docs retrieved for the query by BM25
        """
        start_time = time.time()
        bm25_faiss_index = faiss.read_index(os.path.join(output_path, "bm25_faiss_{}.index".format(fold)))
        doc_id_to_faiss_id_fn = os.path.join(output_path, "doc_id_to_faiss_id_{}.dump".format(fold))
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
                doc_emb = bm25_faiss_index.reconstruct(faiss_id).reshape(768)
                score = np.dot(query_emb, doc_emb)
                score = score.astype(np.float16).item()
                results[qid].append((score, doc_id))

        for qid in results:
            score_doc_id = results[qid]
            for i, (score, doc_id) in enumerate(sorted(score_doc_id, reverse=True)):
                run.setdefault(qid, {})[doc_id] = score

        if hasattr(self.benchmark, "need_pooling") and self.benchmark.need_pooling:
            run = pool_trec_passage_run(run, strategy=self.benchmark.config["pool"])

        metrics = evaluator.eval_runs(run, self.benchmark.qrels, evaluator.DEFAULT_METRICS, self.benchmark.relevance_level)

        faiss_logger.info("Manual search took".format(time.time() - start_time))
        return metrics
