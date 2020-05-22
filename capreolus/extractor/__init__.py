import math
import pickle
from collections import defaultdict
import tensorflow as tf

import os
import numpy as np
import hashlib
from pymagnitude import Magnitude, MagnitudeUtils
from tqdm import tqdm

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, CACHE_BASE_PATH
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError

logger = get_logger(__name__)


class Extractor(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "extractor"

    def _extend_stoi(self, toks_list, calc_idf=False):
        if not self.stoi:
            logger.warning("extending stoi while it's not yet instantiated")
            self.stoi = {}
        # TODO is this warning working correctly?
        if calc_idf and not self.idf:
            logger.warning("extending idf while it's not yet instantiated")
            self.idf = {}
        if calc_idf and not self.modules.get("index", None):
            logger.warning("requesting calculating idf yet index is not available, set calc_idf to False")
            calc_idf = False

        n_words_before = len(self.stoi)
        for toks in toks_list:
            toks = [toks] if isinstance(toks, str) else toks
            for tok in toks:
                if tok not in self.stoi:
                    self.stoi[tok] = len(self.stoi)
                if calc_idf and tok not in self.idf:
                    self.idf[tok] = self["index"].get_idf(tok)

        logger.debug(f"added {len(self.stoi)-n_words_before} terms to the stoi of extractor {self.name}")

    def cache_state(self, qids, docids):
        raise NotImplementedError

    def load_state(self, qids, docids):
        raise NotImplementedError

    def get_state_cache_file_path(self, qids, docids):
        """
        Returns the path to the cache file used to store the extractor state, regardless of whether it exists or not
        """
        sorted_qids = sorted(qids)
        sorted_docids = sorted(docids)
        return self.get_cache_path() / hashlib.md5(str(sorted_qids + sorted_docids).encode("utf-8")).hexdigest()

    def is_state_cached(self, qids, docids):
        """
        Returns a boolean indicating whether the state corresponding to the qids and docids passed has already
        been cached
        """
        return os.path.exists(self.get_state_cache_file_path(qids, docids))

    def _build_vocab(self, qids, docids, topics):
        raise NotImplementedError


class EmbedText(Extractor):
    name = "embedtext"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="anserini"),
    }

    pad = 0
    pad_tok = "<pad>"
    embed_paths = {
        "glove6b": "glove/light/glove.6B.300d",
        "glove6b.50d": "glove/light/glove.6B.50d",
        "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
        "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
    }

    @staticmethod
    def config():
        embeddings = "glove6b"
        zerounk = False
        calcidf = True
        maxqlen = 4
        maxdoclen = 800
        usecache = False

    def _get_pretrained_emb(self):
        magnitude_cache = CACHE_BASE_PATH / "magnitude/"
        return Magnitude(MagnitudeUtils.download_model(self.embed_paths[self.cfg["embeddings"]], download_dir=magnitude_cache))

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2toks = state_dict["docid2toks"]
            self.stoi = state_dict["stoi"]
            self.itos = state_dict["itos"]

        logger.info("Vocabulary loaded from cache")

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "docid2toks": self.docid2toks, "stoi": self.stoi, "itos": self.itos}
            pickle.dump(state_dict, f, protocol=-1)

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
        else:
            tokenize = self["tokenizer"].tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
            self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
            self._extend_stoi(self.qid2toks.values(), calc_idf=self.cfg["calcidf"])
            self._extend_stoi(self.docid2toks.values(), calc_idf=self.cfg["calcidf"])
            self.itos = {i: s for s, i in self.stoi.items()}
            logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")
            self.cache_state(qids, docids)

    def _get_idf(self, toks):
        return [self.idf.get(tok, 0) for tok in toks]

    def _build_embedding_matrix(self):
        assert len(self.stoi) > 1  # needs more vocab than self.pad_tok

        magnitude_emb = self._get_pretrained_emb()
        emb_dim = magnitude_emb.dim
        embed_vocab = set(term for term, _ in magnitude_emb)
        embed_matrix = np.zeros((len(self.stoi), emb_dim), dtype=np.float32)

        n_missed = 0
        for term, idx in tqdm(self.stoi.items()):
            if term in embed_vocab:
                embed_matrix[idx] = magnitude_emb.query(term)
            elif term == self.pad_tok:
                embed_matrix[idx] = np.zeros(emb_dim)
            else:
                n_missed += 1
                embed_matrix[idx] = np.zeros(emb_dim) if self.cfg["zerounk"] else np.random.normal(scale=0.5, size=emb_dim)

        logger.info(f"embedding matrix {self.cfg['embeddings']} constructed, with shape {embed_matrix.shape}")
        if n_missed > 0:
            logger.warning(f"{n_missed}/{len(self.stoi)} (%.3f) term missed" % (n_missed / len(self.stoi)))

        self.embeddings = embed_matrix

    def exist(self):
        return (
            hasattr(self, "embeddings")
            and self.embeddings is not None
            and isinstance(self.embeddings, np.ndarray)
            and 0 < len(self.stoi) == self.embeddings.shape[0]
        )

    def build_query_embedding_matrix(self, qids, docids, topics):
        magnitude_emb = self._get_pretrained_emb()
        emb_dim = magnitude_emb.dim
        embed_vocab = set(term for term, _ in magnitude_emb)
        query_embed_matrix = np.zeros(len(qids), self.cfg["maxqlen"], emb_dim)
        doc_embed_matrix = np.zeros(len(docids), self.cfg["maxdoclen", emb_dim])

        n_missed = 0
        for qid, idx in self.qid2idx.items():
            query_terms = padlist(self["tokenizer"].tokenize(topics[qid]), self.cfg["maxqlen"], pad_token=self.pad_tok)
            query_term_embeds = []
            for term in query_terms:
                if term in embed_vocab:
                    term_embed = magnitude_emb.query(term)
                elif term == self.pad_tok:
                    term_embed = np.zeros(emb_dim)
                else:
                    n_missed += 1
                    term_embed = np.zeros(emb_dim) if self.cfg["zerounk"] else np.random.normal(scale=0.5,
                                                                                                   size=emb_dim)
                query_term_embeds.append(term_embed)

            query_embed = np.stack(query_term_embeds)
            query_embed_matrix[idx] = query_embed

        for docid, idx in self.docid2idx.items():
            doc_terms = padlist(self["tokenizer"].tokenize(self["index"].get_doc(docid)), self.cfg["maxdoclen"], pad_token=self.pad_tok)
            doc_term_embeds = []
            for term in doc_terms:
                if term in embed_vocab:
                    term_embed = magnitude_emb.query(term)
                elif term == self.pad_tok:
                    term_embed = np.zeros(emb_dim)
                else:
                    n_missed += 1
                    term_embed = np.zeros(emb_dim) if self.cfg["zerounk"] else np.random.normal(scale=0.5,
                                                                                                   size=emb_dim)
                doc_term_embeds.append(term_embed)
            doc_embed = np.stack(doc_term_embeds)
            doc_embed_matrix[idx] = doc_embed

    def create(self, qids, docids, topics):

        if self.exist():
            return

        self["index"].create_index()

        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.qid2idx = {}
        self.docid2idx = {}
        self.qid2toks = defaultdict(list)
        self.docid2toks = defaultdict(list)
        self.idf = defaultdict(lambda: 0)
        self.embeddings = None
        # self.cache = self.load_cache()    # TODO

        self._build_vocab(qids, docids, topics)
        self._build_embedding_matrix()

    def _tok2vec(self, toks):
        # return [self.embeddings[self.stoi[tok]] for tok in toks]
        return [self.stoi[tok] for tok in toks]

    def id2vec(self, qid, posid, negid=None):
        query = self.qid2toks[qid]

        # TODO find a way to calculate qlen/doclen stats earlier, so we can log them and check sanity of our values
        qlen, doclen = self.cfg["maxqlen"], self.cfg["maxdoclen"]
        posdoc = self.docid2toks.get(posid, None)
        if not posdoc:
            raise MissingDocError(qid, posid)

        idfs = padlist(self._get_idf(query), qlen, 0)
        query = self._tok2vec(padlist(query, qlen, self.pad_tok))
        posdoc = self._tok2vec(padlist(posdoc, doclen, self.pad_tok))

        # TODO determine whether pin_memory is happening. may not be because we don't place the strings in a np or torch object
        data = {
            "qid": qid,
            "posdocid": posid,
            "idfs": np.array(idfs, dtype=np.float32),
            "query": np.array(query, dtype=np.long),
            "posdoc": np.array(posdoc, dtype=np.long),
            "query_idf": np.array(idfs, dtype=np.float32),
            "negdocid": "",
            "negdoc": np.zeros(self.cfg["maxdoclen"], dtype=np.long),
        }

        if negid:
            negdoc = self.docid2toks.get(negid, None)
            if not negdoc:
                raise MissingDocError(qid, negid)

            negdoc = self._tok2vec(padlist(negdoc, doclen, self.pad_tok))
            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data


class TFEmbedText(Extractor):
    name = "tfembedtext"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="anserini"),
    }

    pad = 0
    pad_tok = "<pad>"
    embed_paths = {
        "glove6b": "glove/light/glove.6B.300d",
        "glove6b.50d": "glove/light/glove.6B.50d",
        "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
        "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
    }

    @staticmethod
    def config():
        embeddings = "glove6b"
        calcidf = True
        maxqlen = 4
        maxdoclen = 800
        usecache = False

    def _get_pretrained_emb(self):
        magnitude_cache = CACHE_BASE_PATH / "magnitude/"
        return Magnitude(
            MagnitudeUtils.download_model(self.embed_paths[self.cfg["embeddings"]], download_dir=magnitude_cache))

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.docid2idx = state_dict["docid2idx"]
            self.qid2idx = state_dict["qid2idx"]
            self.query_embeddings = state_dict["query_embeddings"]
            self.doc_embeddings = state_dict["doc_embeddings"]

        logger.info("Embeddings loaded from cache")

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {
                "docid2idx": self.docid2idx,
                "qid2idx": self.qid2idx,
                "query_embeddings": self.query_embeddings,
                "doc_embeddings": self.doc_embeddings
            }
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "query": tf.io.FixedLenFeature([1], tf.int64),
            "posdoc": tf.io.FixedLenFeature([1], tf.int64),
            "negdoc": tf.io.FixedLenFeature([1], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32, default_value=tf.zeros((1))),
        }

        return feature_description

    def create_tf_feature(self, sample):
        """
        sample - output from self.id2vec()
        return - a tensorflow feature
        """
        query, posdoc, negdoc = sample["query"], sample["posdoc"], sample["negdoc"]
        feature = {
            "query": tf.train.Feature(int64_list=tf.train.Int64List(value=query)),
            "posdoc": tf.train.Feature(int64_list=tf.train.Int64List(value=posdoc)),
            "negdoc": tf.train.Feature(int64_list=tf.train.Int64List(value=negdoc)),
        }

        return feature

    def parse_tf_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)
        posdoc = parsed_example["posdoc"]
        negdoc = parsed_example["negdoc"]
        query = parsed_example["query"]
        label = parsed_example["label"]

        return (posdoc, negdoc, query), label

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
        else:
            # So that the generated indices would be deterministic
            qids = sorted(qids)
            docids = sorted(docids)

            self.qid2idx = {qid: idx + 1 for idx, qid in enumerate(qids)}
            self.qid2idx['empty_query'] = 0
            self.docid2idx = {docid: idx+1 for idx, docid in enumerate(docids)}
            self.docid2idx['empty_doc'] = 0  # Special doc that is just zeros
            self.query_embeddings = self.build_embedding_matrix(self.qid2idx, self.cfg["maxqlen"], topics.get)
            self.doc_embeddings = self.build_embedding_matrix(self.docid2idx, self.cfg["maxdoclen"], self["index"].get_doc)
            self.cache_state(qids, docids)

    def build_embedding_matrix(self, id2idx, doclen, getter):
        magnitude_emb = self._get_pretrained_emb()
        embed_vocab = set(term for term, _ in magnitude_emb)
        embed_matrix = np.zeros((len(id2idx), doclen, magnitude_emb.emb_dim), dtype=np.float)

        embed_matrix[0] = np.zeros((doclen, magnitude_emb.emb_dim), dtype=np.float)

        for record_id, idx in tqdm(id2idx.items(), desc="Embedding matrix"):
            if idx == 0:
                # The zero index is reserved for an empty record. Handle it separately
                continue

            terms = padlist(self["tokenizer"].tokenize(getter(record_id)), doclen, pad_token=self.pad_tok)
            term_embeds = [self.get_term_embed(term, embed_vocab, magnitude_emb) for term in terms]
            record_embed = np.stack(term_embeds)
            embed_matrix[idx] = record_embed

        # Cost of each value in the matrix in megabytes, assume float32
        unit_cost = (doclen * magnitude_emb.emb_dim * 32) / (1024 * 1024)
        logger.info("Unit cost is {}".format(unit_cost))
        # Each tensor should be less than 2GB in memory
        limit = 2000

        # Make a list of matrices, each less than 2GB in memory.
        # tf.python.ops.embedding_ops.embedding_lookup() can make use of shards
        step = math.floor(limit / unit_cost)
        logger.info("{} indices would fit into a single matrix".format(step))
        matrices = []
        for i in range(0, len(id2idx), step):
            matrices.append(tf.identity(embed_matrix[i:i+step]))

        logger.info("Number of shards: {}".format(len(matrices)))

        return matrices

    def get_term_embed(self, term, embed_vocab, magnitude_emb):
        emb_dim = magnitude_emb.emb_dim
        if term in embed_vocab:
            term_embed = magnitude_emb.query(term)
        elif term == self.pad_tok:
            term_embed = np.zeros(emb_dim)
        else:
            # Warning: Do not set unknown terms as np.zero - that would break similarity matrix calculation
            term_embed = np.random.normal(scale=0.5, size=emb_dim)

        return term_embed

    def exist(self):
        return (
            hasattr(self, "query_embeddings")
            and hasattr(self, "doc_embeddings")
            and self.query_embeddings is not None
            and self.doc_embeddings is not None
        )

    def create(self, qids, docids, topics):
        if self.exist():
            return

        self["index"].create_index()

        self.qid2idx = {}
        self.docid2idx = {}
        self.query_embeddings = None
        self.doc_embeddings = None

        self._build_vocab(qids, docids, topics)

    def id2vec(self, qid, posid, negid=None):
        data = {
            "query": np.array([self.qid2idx[qid]], dtype=np.long),
            "posdoc": np.array([self.docid2idx[posid]], dtype=np.long),
            "negdoc": np.array([0], dtype=np.long)
        }

        if negid:
            data["negdoc"] = np.array([self.docid2idx[negid]], dtype=np.long)

        return data


class BertText(Extractor):
    name = "berttext"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="berttokenizer"),
    }

    pad = 0
    pad_tok = "<pad>"

    @staticmethod
    def config():
        maxqlen = 4
        maxdoclen = 800
        usecache = False

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2toks = state_dict["docid2toks"]
            self.clsidx = state_dict["clsidx"]
            self.sepidx = state_dict["sepidx"]

        logger.info("Vocabulary loaded from cache")

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "docid2toks": self.docid2toks, "clsidx": self.clsidx, "sepidx": self.sepidx}
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "query": tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.int64),
            "query_mask": tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.int64),
            "posdoc": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "posdoc_mask": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "negdoc": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "negdoc_mask": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32, default_value=tf.zeros((1))),
        }

        return feature_description

    def create_tf_feature(self, sample):
        """
        sample - output from self.id2vec()
        return - a tensorflow feature
        """
        query, posdoc, negdoc, negdoc_id = sample["query"], sample["posdoc"], sample["negdoc"], sample["negdocid"]
        query_mask, posdoc_mask, negdoc_mask = sample["query_mask"], sample["posdoc_mask"], sample["negdoc_mask"]

        feature = {
            "query": tf.train.Feature(int64_list=tf.train.Int64List(value=query)),
            "query_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=query_mask)),
            "posdoc": tf.train.Feature(int64_list=tf.train.Int64List(value=posdoc)),
            "posdoc_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=posdoc_mask)),
            "negdoc": tf.train.Feature(int64_list=tf.train.Int64List(value=negdoc)),
            "negdoc_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=negdoc_mask)),
        }

        return feature

    def parse_tf_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)
        posdoc = parsed_example["posdoc"]
        posdoc_mask = parsed_example["posdoc_mask"]
        negdoc = parsed_example["negdoc"]
        negdoc_mask = parsed_example["negdoc_mask"]
        query = parsed_example["query"]
        query_mask = parsed_example["query_mask"]
        label = parsed_example["label"]

        return (posdoc, posdoc_mask, negdoc, negdoc_mask, query, query_mask), label

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
        else:
            logger.info("Building bertext vocabulary")
            tokenize = self["tokenizer"].tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in tqdm(docids, desc="doctoks")}
            self.clsidx, self.sepidx = self["tokenizer"].convert_tokens_to_ids(["CLS", "SEP"])

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "docid2toks") and len(self.docid2toks)

    def create(self, qids, docids, topics):
        if self.exist():
            return

        self["index"].create_index()
        self.qid2toks = defaultdict(list)
        self.docid2toks = defaultdict(list)
        self.clsidx = None
        self.sepidx = None

        self._build_vocab(qids, docids, topics)

    def id2vec(self, qid, posid, negid=None):
        tokenizer = self["tokenizer"]
        qlen, doclen = self.cfg["maxqlen"], self.cfg["maxdoclen"]

        query_toks = tokenizer.convert_tokens_to_ids(self.qid2toks[qid])
        query_mask = self.get_mask(query_toks, qlen)
        query = padlist(query_toks, qlen)

        posdoc_toks = tokenizer.convert_tokens_to_ids(self.docid2toks[posid])
        posdoc_mask = self.get_mask(posdoc_toks, doclen)
        posdoc = padlist(posdoc_toks, doclen)

        data = {
            "qid": qid,
            "posdocid": posid,
            "idfs": np.zeros(qlen, dtype=np.float32),
            "query": np.array(query, dtype=np.long),
            "query_mask": np.array(query_mask, dtype=np.long),
            "posdoc": np.array(posdoc, dtype=np.long),
            "posdoc_mask": np.array(posdoc_mask, dtype=np.long),
            "query_idf": np.array(query, dtype=np.float32),
            "negdocid": "",
            "negdoc": np.zeros(doclen, dtype=np.long),
            "negdoc_mask": np.zeros(doclen, dtype=np.long),
        }

        if negid:
            negdoc_toks = tokenizer.convert_tokens_to_ids(self.docid2toks.get(negid, None))
            negdoc_mask = self.get_mask(negdoc_toks, doclen)
            negdoc = padlist(negdoc_toks, doclen)

            if not negdoc:
                raise MissingDocError(qid, negid)

            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)
            data["negdoc_mask"] = np.array(negdoc_mask, dtype=np.long)

        return data

    def get_mask(self, doc, to_len):
        """
        Returns a mask where it is 1 for actual toks and 0 for pad toks
        """
        s = doc[:to_len]
        padlen = to_len - len(s)
        mask = [1 for _ in s] + [0 for _ in range(padlen)]
        return mask
