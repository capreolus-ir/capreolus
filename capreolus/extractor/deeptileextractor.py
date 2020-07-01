import math
import os
import pickle
import re
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
from nltk import TextTilingTokenizer
from profane import ConfigOption, Dependency, constants
from pymagnitude import Magnitude, MagnitudeUtils
from tqdm import tqdm

from capreolus.utils.common import padlist
from capreolus.utils.loginit import get_logger

from . import Extractor

logger = get_logger(__name__)
CACHE_BASE_PATH = constants["CACHE_BASE_PATH"]


@Extractor.register
class DeepTileExtractor(Extractor):
    """ Creates a text tiling matrix. Used by the DeepTileBars reranker. """

    module_name = "deeptiles"
    pad = 0
    pad_tok = "<pad>"

    embed_paths = {
        "glove6b": "glove/light/glove.6B.300d",
        "glove6b.50d": "glove/light/glove.6B.50d",
        "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
        "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
    }

    requires_random_seed = True
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="anserini"),
    ]
    config_spec = [
        ConfigOption("tfchannel", True, "include TF as a channel"),
        ConfigOption("slicelen", 20),
        ConfigOption("keepstops", False, "include stopwords"),
        ConfigOption("tilechannels", 3),
        ConfigOption("embeddings", "glove6b"),
        ConfigOption("passagelen", 20),
        ConfigOption("maxqlen", 4),
        ConfigOption("maxdoclen", 800),
        ConfigOption("usecache", True),
    ]

    def _get_pretrained_emb(self):
        magnitude_cache = CACHE_BASE_PATH / "magnitude/"
        return Magnitude(MagnitudeUtils.download_model(self.embed_paths[self.config["embeddings"]], download_dir=magnitude_cache))

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2toks = state_dict["docid2toks"]
            self.stoi = state_dict["stoi"]
            self.itos = state_dict["itos"]
            self.docid2segments = state_dict["docid2segments"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {
                "qid2toks": self.qid2toks,
                "docid2toks": self.docid2toks,
                "stoi": self.stoi,
                "itos": self.itos,
                "docid2segments": self.docid2segments,
            }
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        raise NotImplementedError()

    def create_tf_feature(self):
        raise NotImplementedError()

    def parse_tf_example(self, example_proto):
        raise NotImplementedError()

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

            if self.config["tfchannel"]:
                channels.append(tf)

            channels.append(self.idf.get(q_tok, 0) if tf else 0)
            sim = max(
                self.gaussian(
                    embeddings_matrix[self.stoi.get(segment_toks[i], self.pad)], embeddings_matrix[self.stoi.get(q_tok, self.pad)]
                )
                if segment_toks[i] != self.pad_tok
                else 0
                for i in range(len(segment_toks))
            )
            channels.append(sim)
        else:
            channels = [0.0] * self.config["tilechannels"]

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
        q_len = self.config["maxqlen"]
        p_len = self.config["passagelen"]
        # The 'document_segments' arg to the method is a list of segments (segregated by topic) in a single document
        # Hence query_to_doc_tiles matrix stores the tiles (scores) b/w each query tok and each passage in the doc
        query_to_doc_tiles = torch.zeros(1, q_len, p_len, self.config["tilechannels"]).float()

        for q_idx in range(q_len):
            q_tok = query_toks[q_idx]
            for seg_idx in range(p_len):
                topic_segment = document_segments[seg_idx]
                tile = self.color_grid(q_tok, topic_segment, embeddings_matrix)
                query_to_doc_tiles[0][q_idx][seg_idx] = tile

        return query_to_doc_tiles

    def _build_embedding_matrix(self):
        magnitude_embeddings = self._get_pretrained_emb()
        embedding_vocab = set(term for term, _ in magnitude_embeddings)

        embedding_matrix = np.zeros((len(self.stoi), magnitude_embeddings.dim), dtype=np.float32)
        for term, idx in tqdm(self.stoi.items(), desc="Building embedding matrix"):
            if term in embedding_vocab:
                embedding_matrix[idx] = magnitude_embeddings.query(term)
            elif term == self.pad_tok:
                padidx = idx
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.5, size=magnitude_embeddings.dim)
            embedding_matrix[padidx] = np.zeros(magnitude_embeddings.dim)

        self.embeddings = embedding_matrix

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            tokenize = self.tokenizer.tokenize
            ttt = TextTilingTokenizer(k=6)  # TODO: Make K configurable?

            # TODO: Move the stoi and itos creation to a reusable mixin
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
            self.docid2toks = {docid: tokenize(self.index.get_doc(docid)) for docid in docids}
            self._extend_stoi(self.qid2toks.values(), calc_idf=True)
            self._extend_stoi(self.docid2toks.values(), calc_idf=True)
            self.itos = {i: s for s, i in self.stoi.items()}
            self.docid2segments = {
                doc_id: self.clean_segments(self.extract_segment(doc_toks, ttt, slicelen=self.config["slicelen"]))
                for doc_id, doc_toks in tqdm(self.docid2toks.items(), desc="Extracting segments")
            }
            if self.config["usecache"]:
                self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "embeddings") and self.embeddings is not None and len(self.stoi) > 0

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()
        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.idf = defaultdict(lambda: 0)
        self.qid2toks = defaultdict(list)
        self.docid2toks = defaultdict(list)
        self.docid2segments = {}
        self.embeddings = None

        self._build_vocab(qids, docids, topics)
        self._build_embedding_matrix()

    def id2vec(self, qid, posdocid, negdocid=None):
        query_toks = padlist(self.qid2toks[qid], self.config["maxqlen"], pad_token=self.pad_tok)
        posdoc_tilebar = self.create_visualization_matrix(query_toks, self.docid2segments[posdocid], self.embeddings)

        data = {
            "qid": qid,
            "query_idf": np.zeros(self.config["maxqlen"], dtype=np.float32),
            "posdocid": posdocid,
            "posdoc": posdoc_tilebar,
            "negdocid": "",
            "negdoc": np.zeros_like(posdoc_tilebar),
        }

        if negdocid:
            negdoc_tilebar = self.create_visualization_matrix(query_toks, self.docid2segments[negdocid], self.embeddings)
            data["negdocid"] = negdocid
            data["negdoc"] = negdoc_tilebar

        return data
