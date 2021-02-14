import os
import torch
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
faiss_logger = get_logger("faiss")


@Extractor.register
class BertText(Extractor):
    """
    Returns documents in the format [CLS] doc tokens [SEP]
    N.B: Right now works only with searcher=faiss. Cannot be used with the regular re-ranking pipeline. See bertpassage.py if you are looking for something that works with a reranker
    """
    module_name = "berttext"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]
    config_spec = [ConfigOption("maxqlen", 4), ConfigOption("maxdoclen", 800), ConfigOption("usecache", False)]

    pad = 0
    pad_tok = "[PAD]"

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.clsidx = state_dict["clsidx"]
            self.sepidx = state_dict["sepidx"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "clsidx": self.clsidx, "sepidx": self.sepidx}
            pickle.dump(state_dict, f, protocol=-1)

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            logger.info("Building bertext vocabulary")
            tokenize = self.tokenizer.tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.clsidx, self.sepidx = self.tokenizer.convert_tokens_to_ids(["CLS", "SEP"])

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "qid2toks") and len(self.qid2toks)

    def preprocess(self, qids, docids, topics):

        self.index.create_index()
        self.qid2toks = defaultdict(list)
        self.clsidx = None
        self.sepidx = None

        self._build_vocab(qids, docids, topics)

    def get_tokenized_doc(self, doc_id):
        doc = self.index.get_doc(doc_id)

        return self.tokenizer.tokenize(doc)

    def id2vec(self, qid, posid, negid=None, label=None):
        assert posid is not None
        tokenizer = self.tokenizer

        posdoc_toks = self.get_tokenized_doc(posid)[:510]
        posdoc_toks = ["[CLS]"] + posdoc_toks + ["[SEP]"]
        posdoc = tokenizer.convert_tokens_to_ids(posdoc_toks)
        posdoc = padlist(posdoc, 512, 0)

        # faiss_logger.debug("Posdocid: {}, doctoks: {}".format(posid, posdoc_toks))
        # faiss_logger.debug("Numericalized posdoc: {}".format(posdoc))
        data = {
            "posdocid": posid,
            "posdoc": np.array(posdoc, dtype=np.long),
            "posdoc_mask": self.get_mask(posdoc)
        }

        if qid:
            query_toks = self.qid2toks[qid][:510]
            query_toks = ["[CLS]"] + query_toks + ["[SEP]"]
            query = tokenizer.convert_tokens_to_ids(query_toks)
            query = padlist(query, 512, 0)
            data["qid"] = qid
            data["query"] = np.array(query, dtype=np.long)
            data["query_mask"] = self.get_mask(query)
            # faiss_logger.debug("qid: {}, query toks: {}".format(qid, query_toks))
            # faiss_logger.debug("Numericalized query: {}".format(query))

        if negid:
            negdoc_toks = self.get_tokenized_doc(negid)[:510]
            negdoc_toks = ["[CLS]"] + negdoc_toks + ["[SEP]"]
            negdoc = tokenizer.convert_tokens_to_ids(negdoc_toks)
            negdoc = padlist(negdoc, 512, 0)

            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)
            data["negdoc_mask"] = self.get_mask(negdoc)
            # faiss_logger.debug("neg docid: {}, doctoks: {}".format(negid, negdoc_toks))
            # faiss_logger.debug("Numericalized_doc: {}".format(negdoc))

        return data

    def id2vec_for_train(self, qid, posid, negid=None, label=None, reldocs=None):
        assert posid is not None
        assert qid is not None
        assert reldocs is not None

        max_doc_length = 256
        max_query_length = 20
        tokenizer = self.tokenizer

        posdoc_toks = self.get_tokenized_doc(posid)[:510]
        posdoc_toks = ["[CLS]"] + posdoc_toks + ["[SEP]"]
        posdoc = tokenizer.convert_tokens_to_ids(posdoc_toks)[:max_doc_length]

        # faiss_logger.debug("Posdocid: {}, doctoks: {}".format(posid, posdoc_toks))
        # faiss_logger.debug("Numericalized posdoc: {}".format(posdoc))
        data = {
            "posdocid": posid,
            "posdoc": np.array(posdoc, dtype=np.long),
            "rel_docs": reldocs
        }

        query_toks = self.qid2toks[qid][:510]
        query_toks = ["[CLS]"] + query_toks + ["[SEP]"]
        query = tokenizer.convert_tokens_to_ids(query_toks)[:max_query_length]
        data["qid"] = qid
        data["query"] = np.array(query, dtype=np.long)

        return data

    def get_mask(self, numericalized_text):
        """
        Returns a mask where it is 1 for actual toks and 0 for pad toks
        """
        return torch.tensor([1 if t !=0 else 0 for t in numericalized_text], dtype=torch.long)
