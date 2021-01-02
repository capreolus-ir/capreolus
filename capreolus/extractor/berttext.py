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

    def id2vec(self, qid, posid, negid=None, label=None):
        assert posid is not None
        tokenizer = self.tokenizer
        data = {}

        posdoc_toks = self.docid2toks[posid][:510]
        posdoc_toks = ["[CLS]"] + posdoc_toks + ["[SEP]"]
        posdoc = tokenizer.convert_tokens_to_ids(posdoc_toks)

        data = {
            "posdocid": posid,
            "posdoc": np.array(posdoc, dtype=np.long),
        }

        if qid:
            query_toks = self.qid2toks[qid][:510]
            query_toks = ["[CLS]"] + query_toks + ["[SEP]"]
            query = tokenizer.convert_tokens_to_ids(query_toks)
            data["qid"] = qid
            data["query"] = np.array(query, dtype=np.long)

        if negid:
            negdoc_toks = self.docid2toks[negid][:510]
            negdoc_toks = ["[CLS]"] + negdoc_toks + ["[SEP]"]
            negdoc = tokenizer.convert_tokens_to_ids(negdoc_toks)

            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data

