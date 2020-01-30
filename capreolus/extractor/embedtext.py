import csv
import os
from collections import defaultdict

import capnp
import numpy as np
import torch
from tqdm import tqdm
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.tokenizer import Tokenizer
from capreolus.extractor import Extractor, BuildStoIMixin
from capreolus.utils.common import padlist
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class EmbedText(Extractor, BuildStoIMixin):
    """ Creates a similarity matrix. Used by similarity matrix-based rankers like KNRM and PACRR. Also provides IDF. """

    pad = 0
    pad_tok = "<pad>"
    tokenizer_name = "anserini"
    embedding_lookup = {
        "glove6b": "glove/light/glove.6B.300d",
        "glove6b.50d": "glove/light/glove.6B.50d",
        "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
        "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
    }

    def __init__(self, *args, **kwargs):
        super(EmbedText, self).__init__(*args, **kwargs)
        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.idf = defaultdict(lambda: 0)
        self.doc_id_to_doc_toks = {}

    @staticmethod
    def config():
        embeddings = "glove6b"  # static embedding file to use: glove6b, glove6b.50d, w2vnews, or fasttext
        keepstops = False  # keep stopwords? discard them if False
        return locals().copy()  # ignored by sacred

    def get_magnitude_embeddings(self, embedding_name):
        return Magnitude(MagnitudeUtils.download_model(self.embedding_lookup[embedding_name], download_dir=self.cache_path))

    def create_embedding_matrix(self, embedding_name):
        zerounk = self.pipeline_config["reranker"] in ["CDSSM"]
        logger.debug("zerounk: %s", zerounk)

        logger.debug("loading %s from pymagnitude", embedding_name)
        magnitude_embeddings = self.get_magnitude_embeddings(embedding_name)

        embedding_vocab = set(term for term, _ in magnitude_embeddings)

        embedding_matrix = np.zeros((len(self.stoi), magnitude_embeddings.dim), dtype=np.float32)
        for term, idx in self.stoi.items():
            if term in embedding_vocab:
                embedding_matrix[idx] = magnitude_embeddings.query(term)
            elif term == "<pad>":
                padidx = idx
            else:
                if zerounk:
                    embedding_matrix[idx] = np.zeros(magnitude_embeddings.dim)
                else:
                    embedding_matrix[idx] = np.random.normal(scale=0.5, size=magnitude_embeddings.dim)
            embedding_matrix[padidx] = np.zeros(magnitude_embeddings.dim)
        return embedding_matrix

    def build_from_benchmark(self, embeddings, keepstops):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index.")

        # tokenizer = Tokenizer.ALL[self.tokenizer_name](self.index, keepstops=keepstops)
        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(self.index, keepstops=keepstops)
        tokenizer.create()
        self.index.open()

        qids = set(self.benchmark.pred_pairs.keys()).union(self.benchmark.train_pairs.keys())
        doc_ids = set()
        for qdocs in self.benchmark.pred_pairs.values():
            doc_ids.update(qdocs)
        for qdocs in self.benchmark.train_pairs.values():
            doc_ids.update(qdocs)

        queries = self.collection.topics[self.benchmark.query_type]
        query_text_list = [queries[qid] for qid in qids]
        query_tocs_list = [tokenizer.tokenize(query) for query in query_text_list]
        logger.info("loading %s queries", len(queries))
        self.build_stoi(query_tocs_list, keepstops, True)

        # TODO: Print max query size as well. Issue a warning if a query with lenght more than maxqlen is embedded
        logger.debug("query vocabulary size: %s", len(self.stoi))

        self.doc_id_to_doc_toks = tokenizer.tokenizedocs(doc_ids)
        self.build_stoi([toks for doc_id, toks in self.doc_id_to_doc_toks.items()], keepstops, False)

        logger.debug("vocabulary size: %s", len(self.stoi))
        tokenizer.write_cache()
        self.embeddings = self.create_embedding_matrix(embeddings)
        self.itos = {i: s for s, i in self.stoi.items()}

    def transform_qid_posdocid_negdocid(self, q_id, posdoc_id, negdoc_id=None):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index")

        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(
            self.index, keepstops=self.pipeline_config["keepstops"]
        )
        queries = self.collection.topics[self.benchmark.query_type]

        posdoc_toks = self.doc_id_to_doc_toks.get(posdoc_id)
        query_toks = tokenizer.tokenize(queries[q_id])

        if not posdoc_toks:
            logger.debug("missing docid %s", posdoc_id)
            return None
        transformed_query = self.transform_txt(query_toks, self.pipeline_config["maxqlen"])
        idfs = padlist([self.idf[self.itos[tok]] for tok in transformed_query], self.pipeline_config["maxqlen"])
        transformed = {
            "qid": q_id,
            "posdocid": posdoc_id,
            "negdocid": negdoc_id,
            "query": transformed_query,
            "posdoc": self.transform_txt(posdoc_toks, self.pipeline_config["maxdoclen"]),
            "query_idf": np.array(idfs, dtype=np.float32),
        }

        if negdoc_id is not None:
            negdoc_toks = self.doc_id_to_doc_toks.get(negdoc_id)
            if not negdoc_toks:
                logger.debug("missing docid %s", negdoc_id)
                return None

            transformed["negdoc"] = self.transform_txt(negdoc_toks, self.pipeline_config["maxdoclen"])

        return transformed

    def transform_txt(self, txt, maxlen):
        toks = [self.stoi.get(term, 0) for term in txt]
        embed = np.array(padlist(toks, maxlen))
        return embed
