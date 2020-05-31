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

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "docid2toks": self.docid2toks, "stoi": self.stoi, "itos": self.itos}
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "query": tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.int64),
            "query_idf": tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.float32),
            "posdoc": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "negdoc": tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.int64),
            "label": tf.io.FixedLenFeature([2], tf.float32, default_value=tf.convert_to_tensor([1, 0], dtype=tf.float32)),
        }

        return feature_description

    def create_tf_feature(self, sample):
        """
        sample - output from self.id2vec()
        return - a tensorflow feature
        """
        query, query_idf, posdoc, negdoc = (sample["query"], sample["query_idf"], sample["posdoc"], sample["negdoc"])
        feature = {
            "query": tf.train.Feature(int64_list=tf.train.Int64List(value=query)),
            "query_idf": tf.train.Feature(float_list=tf.train.FloatList(value=query_idf)),
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
        query_idf = parsed_example["query_idf"]
        label = parsed_example["label"]

        return (posdoc, negdoc, query, query_idf), label

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            tokenize = self["tokenizer"].tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
            self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
            self._extend_stoi(self.qid2toks.values(), calc_idf=self.cfg["calcidf"])
            self._extend_stoi(self.docid2toks.values(), calc_idf=self.cfg["calcidf"])
            self.itos = {i: s for s, i in self.stoi.items()}
            logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")
            if self.cfg["usecache"]:
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

    def create(self, qids, docids, topics):

        if self.exist():
            return

        self["index"].create_index()

        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
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
            "label": tf.io.FixedLenFeature([2], tf.float32, default_value=tf.convert_to_tensor([1, 0], dtype=tf.float32)),
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
            logger.info("Vocabulary loaded from cache")
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


class BertPassage(Extractor):
    name = "bertpassage"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="berttokenizer"),
    }

    pad = 0
    pad_tok = " "

    @staticmethod
    def config():
        maxseqlen = 256
        numpassages = 16
        maxqlen = 8
        passagelen = 150
        stride = 100
        usecache = False

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2passages = state_dict["docid2passages"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "docid2passages": self.docid2passages}
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "posdoc": tf.io.FixedLenFeature([], tf.string),
            "posdoc_mask": tf.io.FixedLenFeature([], tf.string),
            "posdoc_seg": tf.io.FixedLenFeature([], tf.string),
            "negdoc": tf.io.FixedLenFeature([], tf.string),
            "negdoc_mask": tf.io.FixedLenFeature([], tf.string),
            "negdoc_seg": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([2], tf.float32, default_value=tf.convert_to_tensor([1, 0], dtype=tf.float32)),
        }

        return feature_description

    def create_tf_feature(self, sample):
        """
        sample - output from self.id2vec()
        return - a tensorflow feature
        """
        posdoc, negdoc, negdoc_id = sample["posdoc"], sample["negdoc"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = sample["posdoc_mask"], sample["posdoc_seg"], sample["negdoc_mask"], sample["negdoc_seg"]

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):  # if value ist tensor
                value = value.numpy()  # get value of tensor
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {
            "posdoc": _bytes_feature(tf.io.serialize_tensor(posdoc)),
            "posdoc_mask": _bytes_feature(tf.io.serialize_tensor(posdoc_mask)), 
            "posdoc_seg": _bytes_feature(tf.io.serialize_tensor(posdoc_seg)),
            "negdoc": _bytes_feature(tf.io.serialize_tensor(negdoc)),
            "negdoc_mask": _bytes_feature(tf.io.serialize_tensor(negdoc_mask)), 
            "negdoc_seg": _bytes_feature(tf.io.serialize_tensor(negdoc_seg)),
        }

        return feature

    def parse_tf_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)
        posdoc = tf.io.parse_tensor(parsed_example["posdoc"], tf.string)
        posdoc_mask = tf.io.parse_tensor(parsed_example["posdoc_mask"], tf.string)
        posdoc_seg = tf.io.parse_tensor(parsed_example["posdoc_seg"], tf.string)
        negdoc = tf.io.parse_tensor(parsed_example["negdoc"], tf.string)
        negdoc_mask = tf.io.parse_tensor(parsed_example["negdoc_mask"], tf.string)
        negdoc_seg = tf.io.parse_tensor(parsed_example["negdoc_seg"], tf.string)
        label = parsed_example["label"]

        return (posdoc, posdoc_mask, posdoc_seg, negdoc, negdoc_mask, negdoc_seg), label

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            logger.info("Building bertpassage vocabulary")
            tokenize = self["tokenizer"].tokenize
            get_doc = self["index"].get_doc
            self.docid2passages = {}

            # TODO: Move this to a method
            for docid in tqdm(docids, "extract passages"):
                # Naive tokenization based on white space
                doc = get_doc(docid).split()
                passages = []
                for i in range(0, self.cfg["numpassages"]):
                    if i >= len(doc):
                        passage = padlist([], padlen=self.cfg["passagelen"], pad_token=self.pad_tok)
                    else:
                        passage = padlist(doc[i: i+self.cfg["passagelen"]], padlen=self.cfg["passagelen"], pad_token=self.pad_tok)

                    # N.B: The passages are not bert tokenized.
                    passages.append(passage)

                self.docid2passages[docid] = passages

            self.qid2toks = {qid: tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "docid2passages") and len(self.docid2passages)

    def create(self, qids, docids, topics):
        if self.exist():
            return

        if self.cfg["maxseqlen"] < self.cfg["passagelen"] + self.cfg["maxqlen"] + 3:
            raise ValueError("maxseqlen is too short")

        self["index"].create_index()
        self.qid2toks = defaultdict(list)
        self.docid2passages = None

        self._build_vocab(qids, docids, topics)

    def id2vec(self, qid, posid, negid=None):
        tokenizer = self["tokenizer"]
        tokenize = tokenizer.tokenize

        query_toks = self.qid2toks[qid]
        pos_bert_inputs = []
        pos_bert_masks = []
        pos_bert_segs = []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self.docid2passages[posid]
        for passage in pos_passages:
            tokenized_passage = tokenize(" ".join(passage))
            input_line = ['CLS'] + query_toks + ['SEP'] + tokenized_passage + ['SEP']
            padded_input_line = padlist(input_line, padlen=self.cfg["maxseqlen"], pad_token=self.pad_tok)
            pos_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
            pos_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
            pos_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))


        logger.info("pos_bert_masks are {}".format([len(x) for x in pos_bert_masks]))

        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "posdocid": posid,
            "posdoc": np.array(pos_bert_inputs, dtype=np.long),
            "posdoc_mask": np.array(pos_bert_masks, dtype=np.long),
            "posdoc_seg": np.array(pos_bert_segs, dtype=np.long),
            "negdocid": "",
            "negdoc": np.zeros((self.cfg["numpassages"], self.cfg["passagelen"]), dtype=np.long),
            "negdoc_mask": np.zeros((self.cfg["numpassages"], self.cfg["passagelen"]), dtype=np.long),
            "negdoc_seg": np.zeros((self.cfg["numpassages"], self.cfg["passagelen"]), dtype=np.long),

        }

        if negid:
            neg_bert_inputs = []
            neg_bert_masks = []
            neg_bert_segs = []
            neg_passages = self.docid2passages[negid]
            for passage in neg_passages:
                tokenized_passage = tokenize(" ".join(passage))
                input_line = ['CLS'] + query_toks + ['SEP'] + tokenized_passage + ['SEP']
                padded_input_line = padlist(input_line, padlen=self.cfg["maxseqlen"], pad_token=self.pad_tok)
                neg_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
                neg_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
                neg_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))

            if not neg_bert_inputs:
                raise MissingDocError(qid, negid)

            data["negdocid"] = negid
            data["negdoc"] = np.array(neg_bert_inputs, dtype=np.long)
            data["negdoc_mask"] = np.array(neg_bert_masks, dtype=np.long)
            data["negdoc_seg"] = np.array(neg_bert_segs, dtype=np.long)

        return data
