import os
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from capreolus import ConfigOption, Dependency, get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError
from . import Extractor
from .common import load_pretrained_embeddings

logger = get_logger(__name__)


@Extractor.register
class SlowEmbedText(Extractor):
    module_name = "slowembedtext"
    requires_random_seed = True
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="anserini"),
    ]
    config_spec = [
        ConfigOption("embeddings", "glove6b"),
        ConfigOption("zerounk", False),
        ConfigOption("calcidf", True),
        ConfigOption("maxqlen", 4),
        ConfigOption("maxdoclen", 800),
        ConfigOption("usecache", False),
    ]

    pad = 0
    pad_tok = "<pad>"

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
            "query": tf.io.FixedLenFeature([self.config["maxqlen"]], tf.int64),
            "query_idf": tf.io.FixedLenFeature([self.config["maxqlen"]], tf.float32),
            "posdoc": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
            "negdoc": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
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
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            tokenize = self.tokenizer.tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
            self.docid2toks = {docid: tokenize(self.index.get_doc(docid)) for docid in docids}
            self._extend_stoi(self.qid2toks.values(), calc_idf=self.config["calcidf"])
            self._extend_stoi(self.docid2toks.values(), calc_idf=self.config["calcidf"])
            self.itos = {i: s for s, i in self.stoi.items()}
            logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")
            if self.config["usecache"]:
                self.cache_state(qids, docids)

    def _get_idf(self, toks):
        return [self.idf.get(tok, 0) for tok in toks]

    def _load_pretrained_embeddings(self):
        return load_pretrained_embeddings(self.config["embeddings"])

    def _build_embedding_matrix(self):
        assert len(self.stoi) > 1  # needs more vocab than self.pad_tok

        embeddings, _, embedding_stoi = self._load_pretrained_embeddings()
        emb_dim = embeddings.shape[-1]
        embed_matrix = np.zeros((len(self.stoi), emb_dim), dtype=np.float32)

        n_missed = 0
        for term, idx in tqdm(self.stoi.items()):
            if term in embedding_stoi:
                embed_matrix[idx] = embeddings[embedding_stoi[term]]
            elif term == self.pad_tok:
                embed_matrix[idx] = np.zeros(emb_dim)
            else:
                n_missed += 1
                embed_matrix[idx] = np.zeros(emb_dim) if self.config["zerounk"] else np.random.normal(scale=0.5, size=emb_dim)

        logger.info(f"embedding matrix {self.config['embeddings']} constructed, with shape {embed_matrix.shape}")
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

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()

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
        qlen, doclen = self.config["maxqlen"], self.config["maxdoclen"]
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
            "negdoc": np.zeros(self.config["maxdoclen"], dtype=np.long),
        }

        if negid:
            negdoc = self.docid2toks.get(negid, None)
            if not negdoc:
                raise MissingDocError(qid, negid)

            negdoc = self._tok2vec(padlist(negdoc, doclen, self.pad_tok))
            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data
