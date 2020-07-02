from collections import defaultdict

import numpy as np
import tensorflow as tf

from capreolus import ConfigOption, Dependency, constants, get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError

from . import Extractor
from .common import load_pretrained_embeddings

logger = get_logger(__name__)


@Extractor.register
class EmbedText(Extractor):
    module_name = "embedtext"
    requires_random_seed = True
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="anserini"),
    ]
    config_spec = [
        ConfigOption("embeddings", "glove6b"),
        ConfigOption("calcidf", True),
        ConfigOption("maxqlen", 4),
        ConfigOption("maxdoclen", 800),
    ]

    pad_tok = "<pad>"

    def build(self):
        self._embedding_cache = constants["CACHE_BASE_PATH"] / "embeddings"
        self._numpy_cache = self._embedding_cache / (self.config["embeddings"] + ".npy")
        self._vocab_cache = self._embedding_cache / (self.config["embeddings"] + ".vocab.txt")
        self.embeddings, self.stoi, self.itos = None, None, None
        self._next_oov_index = -1

    def _load_pretrained_embeddings(self):
        if self.embeddings is not None:
            return

        self.embeddings, self.itos, self.stoi = load_pretrained_embeddings(self.config["embeddings"])

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

    def _get_idf(self, toks):
        return [self.idf.get(tok, 0) for tok in toks]

    def preprocess(self, qids, docids, topics):
        self._load_pretrained_embeddings()

        self.index.create_index()

        self.qid2toks = {}
        self.docid2toks = {}
        self.idf = defaultdict(lambda: 0)

        for qid in qids:
            if qid not in self.qid2toks:
                self.qid2toks[qid] = self.tokenizer.tokenize(topics[qid])
                self._add_oov_to_vocab(self.qid2toks[qid])

    def get_doc_tokens(self, docid):
        if docid not in self.docid2toks:
            self.docid2toks[docid] = self.tokenizer.tokenize(self.index.get_doc(docid))
            self._add_oov_to_vocab(self.docid2toks[docid])

        return self.docid2toks[docid]

    def _add_oov_to_vocab(self, tokens):
        for tok in tokens:
            if tok not in self.stoi:
                self.stoi[tok] = self._next_oov_index
                self.itos[self._next_oov_index] = tok
                self._next_oov_index -= 1

    def _tok2vec(self, toks):
        return [self.stoi[tok] for tok in toks]

    def id2vec(self, qid, posid, negid=None):
        query = self.qid2toks[qid]

        # TODO find a way to calculate qlen/doclen stats earlier, so we can log them and check sanity of our values
        qlen, doclen = self.config["maxqlen"], self.config["maxdoclen"]
        posdoc = self.get_doc_tokens(posid)
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
            negdoc = self.get_doc_tokens(negid)
            if not negdoc:
                raise MissingDocError(qid, negid)

            negdoc = self._tok2vec(padlist(negdoc, doclen, self.pad_tok))
            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data
