import math
import os
import pickle
import re
import time
from functools import reduce
import torch

from nltk.tokenize import TextTilingTokenizer
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.extractor import Extractor, BuildStoIMixin
from capreolus.tokenizer import Tokenizer
from capreolus.utils.common import padlist
from capreolus.utils.loginit import get_logger
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict

logger = get_logger(__name__)  # pylint: disable=invalid-name


class DeepTileExtractor(Extractor, BuildStoIMixin):
    """ Creates a text tiling matrix. Used by the DeepTileBars reranker. """

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
        super(DeepTileExtractor, self).__init__(*args, **kwargs)
        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.embeddings = None
        self.idf = defaultdict(lambda: 0)
        self.doc_id_to_doc_toks = {}
        self.embeddings_matrix = None
        self.tilechannels = 3
        if not self.pipeline_config["tfchannel"]:
            self.tilechannels -= 1

    @staticmethod
    def config():
        tfchannel = True
        slicelen = 20
        keepstops = False
        return locals().copy()  # ignored by sacred

    def get_magnitude_embeddings(self, embedding_name):
        return Magnitude(MagnitudeUtils.download_model(self.embedding_lookup[embedding_name], download_dir=self.cache_path))

    def extract_segment(self, doc_toks, ttt, slicelen=20):
        """
        1. Tries to extract segments using nlt.TextTilingTokenizer (instance passed as an arg)
        2. If that fails, simply splits into segments of 20 tokens each
        """
        # Join the tokens by a white space, but after every 20 tokens inside a double newline character
        # 20 tokens is an arbitrary decision.
        slice_count = math.ceil(len(doc_toks) / slicelen)
        tok_slices = [" ".join(doc_toks[i * slicelen : i * slicelen + slicelen]) for i in range(slice_count)]
        doc_text = "\n\n".join(tok_slices)

        try:
            # tokenize() internally converts the doc_text to lowercase and removes non alpha numeric chars before tiling
            # see https://www.nltk.org/_modules/nltk/tokenize/texttiling.html. Hence we don't have to do any
            # preprocessing. However, the returned segments have everything (i.e non-alphanums) preserved.
            segments = ttt.tokenize(doc_text)
            # Remove all paragraph breaks (the ones that were already there and the ones that we inserted) - we don't
            # really need them once ttt is done
            segments = [re.sub("\n\n", " ", segment) for segment in segments]
        except ValueError:
            # TextTilingTokenizer throws an error if the input is too short (eg: less than 100 chars) or if it could not
            # find any paragraphs. In that case, naively split on every artificial paragraph that we inserted
            segments = doc_text.split("\n\n")

        return segments

    def clean_segments(self, segments, p_len=30):
        """
        1. Pad segments if it's too short
        2. If it's too long, collapse the extra text into the last element
        """

        if len(segments) < p_len:
            segments = padlist(list(segments), p_len, pad_token=self.pad_tok)
        elif len(segments) > p_len:
            segments[p_len - 1] = reduce(lambda a, b: a + b, segments[p_len - 1 :])
            segments = segments[:p_len]

        return segments

    def build_stoi_from_segments(self, segments_list, keepstops, calculate_idf):
        for segments in tqdm(segments_list):
            for segment in segments:
                # The input to the text tiler is already tokenized, so can simply split on whitespace and call it a day
                for tok in segment.split(" "):
                    if tok not in self.stoi:
                        self.stoi[tok] = len(self.stoi)

                        if calculate_idf:
                            self.idf[tok] = self.index.getidf(tok)

    def gaussian(self, x1, z1):
        x = np.asarray(x1)
        z = np.asarray(z1)
        return np.exp((-(np.linalg.norm(x - z) ** 2)) / (2 * 1))

    def color_grid(self, q_tok, topic_segment, embeddings_matrix):
        """
        See the section titles "Coloring" in the original paper: https://arxiv.org/pdf/1811.00606.pdf
        Calculates TF, IDF and max gaussian for the given q_tok <> topic_segment pair
        :param q_tok: List of tokens in a query
        :param topic_segment: A single segment. String. (A document can have multiple segments)
        """
        channels = []
        if q_tok != self.pad_tok and topic_segment != self.pad_tok:
            segment_toks = topic_segment.split(" ")
            tf = segment_toks.count(q_tok)

            if self.pipeline_config["tfchannel"]:
                channels.append(tf)

            channels.append(self.idf.get(q_tok, 0) if tf else 0)
            sim = max(
                self.gaussian(embeddings_matrix[self.stoi[segment_toks[i]]], embeddings_matrix[self.stoi[q_tok]])
                if segment_toks[i] != self.pad_tok
                else 0
                for i in range(len(segment_toks))
            )
            channels.append(sim)
        else:
            channels = [0.0] * self.tilechannels

        tile = torch.tensor(channels, dtype=torch.float)
        return tile

    def create_visualization_matrix(self, query_toks, document_segments, embeddings_matrix):
        """
        Returns a tensor of shape (1, maxqlen, passagelen, channels)
        The first dimension (i.e 1) is dummy. Ignore that
        The 2nd and 3rd dimensions (i.e maxqlen and passagelen) together represents a "tile" between a query token and
        a passage (i.e doc segment). The "tile" is up to dimension 3 - it contains TF of the query term in that passage,
        idf of the query term, and the max word2vec similarity between query term and any term in the passage
        :param query_toks: A list of tokens in the query. Eg: ['hello', 'world']
        :param document_segments: List of segments in a document. Each segment is a string
        :param embeddings_matrix: Used to look up word2vec embeddings
        """
        q_len = self.pipeline_config["maxqlen"]
        p_len = self.pipeline_config["passagelen"]
        # The 'document_segments' arg to the method is a list of segments (segregated by topic) in a single document
        # Hence query_to_doc_tiles matrix stores the tiles (scores) b/w each query tok and each passage in the doc
        query_to_doc_tiles = torch.zeros(1, q_len, p_len, self.tilechannels).float()

        for q_idx in range(q_len):
            q_tok = query_toks[q_idx]
            for seg_idx in range(p_len):
                topic_segment = document_segments[seg_idx]
                tile = self.color_grid(q_tok, topic_segment, embeddings_matrix)
                query_to_doc_tiles[0][q_idx][seg_idx] = tile

        return query_to_doc_tiles

    def create_embedding_matrix(self, embedding_name):
        logger.debug("loading %s from pymagnitude", embedding_name)
        magnitude_embeddings = self.get_magnitude_embeddings(embedding_name)

        embedding_vocab = set(term for term, _ in magnitude_embeddings)

        embedding_matrix = np.zeros((len(self.stoi), magnitude_embeddings.dim), dtype=np.float32)
        for term, idx in self.stoi.items():
            if term in embedding_vocab:
                embedding_matrix[idx] = magnitude_embeddings.query(term)
            elif term == self.pad_tok:
                padidx = idx
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.5, size=magnitude_embeddings.dim)
            embedding_matrix[padidx] = np.zeros(magnitude_embeddings.dim)
        return embedding_matrix

    def build_from_benchmark(self, keepstops, *args, **kwargs):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index.")

        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(self.index, keepstops=keepstops)
        tokenizer.create()
        self.index.open()

        # Tokenize all queries
        qids = list(set(self.benchmark.pred_pairs.keys()).union(set(self.benchmark.train_pairs.keys())))
        queries = self.collection.topics[self.benchmark.query_type]
        query_text_list = [queries[qid] for qid in qids]
        query_tocs_list = [tokenizer.tokenize(query) for query in query_text_list]

        # Tokenize all documents that this benchmark will use
        doc_ids = set()
        for qdocs in self.benchmark.pred_pairs.values():
            doc_ids.update(qdocs)
        for qdocs in self.benchmark.train_pairs.values():
            doc_ids.update(qdocs)
        self.doc_id_to_doc_toks = tokenizer.tokenizedocs(doc_ids)

        logger.debug("Building stoi")
        self.build_stoi(query_tocs_list, keepstops, True)

        logger.debug("Extracting segments")
        ttt = TextTilingTokenizer(k=6)

        logger.debug("cache path: %s", self.feature_cache_dir)
        cache_file = os.path.join(self.feature_cache_dir, "segments.npy")
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        if os.path.isfile(cache_file):
            logger.debug("Using cached segments")
            doc_id_to_segments = pickle.load(open(cache_file, "rb"))
        else:
            doc_id_to_segments = {
                doc_id: self.extract_segment(doc_toks, ttt, slicelen=self.pipeline_config["slicelen"])
                for doc_id, doc_toks in tqdm(self.doc_id_to_doc_toks.items())
            }
            try:
                pickle.dump(doc_id_to_segments, open(cache_file, "wb"), protocol=2)
            except FileNotFoundError as ex:
                logger.error("encountered exception while writing segment cache file %s: %s", cache_file, ex)
                pass

        self.doc_id_to_segments = {doc_id: self.clean_segments(segments) for doc_id, segments in doc_id_to_segments.items()}
        self.build_stoi_from_segments([segments for doc_id, segments in doc_id_to_segments.items()], keepstops, False)
        logger.info("Creating tile bars")
        self.embeddings_matrix = self.create_embedding_matrix("glove6b.50d")
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.debug("The vocabulary size: %s", len(self.stoi))

    def transform_qid_posdocid_negdocid(self, q_id, posdoc_id, negdoc_id=None):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index")

        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(
            self.index, keepstops=self.pipeline_config["keepstops"]
        )
        queries = self.collection.topics[self.benchmark.query_type]
        query_toks = tokenizer.tokenize(queries[q_id])
        #
        query_toks = padlist(query_toks, self.pipeline_config["maxqlen"], pad_token=self.pad_tok)
        posdoc_tilebar = self.create_visualization_matrix(query_toks, self.doc_id_to_segments[posdoc_id], self.embeddings_matrix)
        if negdoc_id:
            negdoc_tilebar = self.create_visualization_matrix(
                query_toks, self.doc_id_to_segments[negdoc_id], self.embeddings_matrix
            )
        else:
            negdoc_tilebar = torch.zeros(
                1, self.pipeline_config["maxqlen"], self.pipeline_config["passagelen"], self.tilechannels
            ).float()

        # All we need are the doc and query ids. See DeepTileBar reranker. Hence setting others to dummy values
        transformed = {
            "qid": q_id,
            "posdocid": posdoc_id,
            "negdocid": negdoc_id,
            "query": np.zeros(1),
            "posdoc": posdoc_tilebar,
            "negdoc": negdoc_tilebar,
            "query_idf": np.zeros(1),
        }

        return transformed
