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

logger = get_logger(__name__)


@Extractor.register
class BertText(Extractor):
    module_name = "berttext"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]
    config_spec = [ConfigOption("maxqlen", 4), ConfigOption("maxdoclen", 800), ConfigOption("usecache", False)]

    pad = 0
    pad_tok = "<pad>"

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
            "query": tf.io.FixedLenFeature([self.config["maxqlen"]], tf.int64),
            "query_mask": tf.io.FixedLenFeature([self.config["maxqlen"]], tf.int64),
            "posdoc": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
            "posdoc_mask": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
            "negdoc": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
            "negdoc_mask": tf.io.FixedLenFeature([self.config["maxdoclen"]], tf.int64),
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
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            logger.info("Building bertext vocabulary")
            tokenize = self.tokenizer.tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.docid2toks = {docid: tokenize(self.index.get_doc(docid)) for docid in tqdm(docids, desc="doctoks")}
            self.clsidx, self.sepidx = self.tokenizer.convert_tokens_to_ids(["CLS", "SEP"])

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "docid2toks") and len(self.docid2toks)

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()
        self.qid2toks = defaultdict(list)
        self.docid2toks = defaultdict(list)
        self.clsidx = None
        self.sepidx = None

        self._build_vocab(qids, docids, topics)

    def id2vec(self, qid, posid, negid=None):
        tokenizer = self.tokenizer
        qlen, doclen = self.config["maxqlen"], self.config["maxdoclen"]

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
