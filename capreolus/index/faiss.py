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
    
    dependencies = [
                       Dependency(key="index", module="index", name="anserini"),
                       Dependency(key="benchmark", module="benchmark")
                   ] + Index.dependencies
    config_spec = [ConfigOption("isclear", False, "Whether the searcher is used with CLEAR.")]

    def exists(self, fold=None):
        return False

    def create_index(self, fold=None):
        raise Exception("This should not have been called")

    def get_results_path(self):
        """Return an absolute path that can be used for storing results.
        The path is a function of the module's config and the configs of its dependencies.
        """

        return constants["RESULTS_BASE_PATH"] / self.get_module_path()

    def create_shard(self, encoder, shard_id, offset, doc_ids, fold):
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
        dataset.prepare(
            doc_ids, None, encoder.extractor, relevance_level=self.benchmark.relevance_level
        )

        BATCH_SIZE = 64
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.model.to(device)
        encoder.model.eval()

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
                with torch.cuda.amp.autocast():
                    doc_emb = encoder.encode_doc(batch["posdoc"], batch["posdoc_mask"]).cpu().numpy()

            faiss_ids_for_batch = np.array(faiss_ids_for_batch, dtype=np.long).reshape(-1, )
            faiss_index.add_with_ids(doc_emb, faiss_ids_for_batch)

        os.makedirs(self.get_cache_path(), exist_ok=True)
        doc_id_to_faiss_id_fn = os.path.join(self.get_cache_path(), "shard_{}_doc_id_to_faiss_id_{}.dump".format(shard_id, fold))
        pickle.dump(doc_id_to_faiss_id, open(doc_id_to_faiss_id_fn, "wb"), protocol=-1)

        faiss_id_to_doc_id_fn = os.path.join(self.get_cache_path(), "shard_{}_faiss_id_to_doc_id_{}.dump".format(shard_id, fold))
        faiss_id_to_doc_id = {faiss_id: doc_id for doc_id, faiss_id in doc_id_to_faiss_id.items()}
        pickle.dump(faiss_id_to_doc_id, open(faiss_id_to_doc_id_fn, "wb"), protocol=-1)

        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(self.get_index_path(), "shard_{}_faiss_{}.index".format(shard_id, fold)))
        faiss_logger.info("Shard: {} written to disk".format(shard_id))

    def reweight_using_bm25_scores(self, distances, results, qid_query, fold, lambda_test=0.5):
        """
        Takes the cosine similarity scores that FAISS produces and re-weights them as specified in the CLEAR paper
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

    def faiss_search(self, topic_vectors, k, qid_query, numshards, docs_per_shard, fold):
        result_heap = faiss.ResultHeap(nq=len(topic_vectors), k=k)

        aggregated_faiss_id_to_doc_id = {}
        aggregated_doc_id_to_faiss_id = {}
        count_map = defaultdict(lambda: 0)
        index_path = self.get_index_path()
        index_cache_path = self.get_cache_path()
        for shard_id in range(numshards):
            offset = shard_id * docs_per_shard
            filename = os.path.join(index_path, "shard_{}_faiss_{}.index".format(shard_id, fold))
            assert os.path.isfile(filename), "shard {} not found".format(filename)
            faiss_shard = faiss.read_index(os.path.join(index_path, filename))
            distances, results = faiss_shard.search(topic_vectors, k)
            results = results + offset
            result_heap.add_result(D=distances, I=results)

            faiss_id_to_doc_id = pickle.load(open(os.path.join(index_cache_path, "shard_{}_faiss_id_to_doc_id_{}.dump".format(shard_id, fold)), "rb"))
            faiss_id_to_doc_id = {faiss_id + offset: doc_id for faiss_id, doc_id in faiss_id_to_doc_id.items()}
            doc_id_to_faiss_id = {doc_id: faiss_id for faiss_id, doc_id in faiss_id_to_doc_id.items()}
            # doc_id_to_faiss_id = pickle.load(open(os.path.join(index_cache_path, "shard_{}_doc_id_to_faiss_id_{}.dump".format(shard_id, fold)), "rb"))
            for faiss_id, doc_id in faiss_id_to_doc_id.items():
                count_map[faiss_id] += 1

            aggregated_faiss_id_to_doc_id.update(faiss_id_to_doc_id)
            aggregated_doc_id_to_faiss_id.update(doc_id_to_faiss_id)

        result_heap.finalize()
        pickle.dump(aggregated_faiss_id_to_doc_id, open(os.path.join(index_cache_path, "faiss_id_to_doc_id_{}.dump".format(fold)), "wb"), protocol=-1)
        pickle.dump(aggregated_doc_id_to_faiss_id, open(os.path.join(index_cache_path, "doc_id_to_faiss_id_{}.dump".format(fold)), "wb"), protocol=-1)

        if self.config["isclear"]:
            faiss_logger.info("Reweighting FAISS scores for CLEAR")
            distances = self.reweight_using_bm25_scores(result_heap.D, result_heap.I, qid_query, fold)

        temp = 0
        for faiss_id, count in count_map.items():
            if count > 1:
                logger.info("{} has count {}".format(faiss_id, count))
                temp += 1
        logger.info("temp is {}".format(temp))
        assert temp == 0

        return result_heap.D, result_heap.I

    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

