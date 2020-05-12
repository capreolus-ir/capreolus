import os
import numpy as np
import pickle
from collections import defaultdict

from capreolus.extractor import Extractor
from capreolus.registry import Dependency
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError


logger = get_logger(__name__)


class Berttext(Extractor):
    name = "bert"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="bert"),
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

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.cfg["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            tokenize = self["tokenizer"].tokenize
            self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
            self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
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

        query = padlist(tokenizer.convert_tokens_to_ids(self.qid2toks[qid]), qlen)
        posdoc = padlist(tokenizer.convert_tokens_to_ids(self.docid2toks[posid]), doclen)

        data = {
            "qid": qid,
            "posdocid": posid,
            "idfs": np.zeros(qlen, dtype=np.float32),
            "query": np.array(query, dtype=np.long),
            "posdoc": np.array(posdoc, dtype=np.long),
            "query_idf": np.array(query, dtype=np.float32),
        }

        if negid:
            negdoc = padlist(tokenizer.convert_tokens_to_ids(self.docid2toks.get(negid, None)), doclen)
            if not negdoc:
                raise MissingDocError(qid, negid)

            data["negdocid"] = negid
            data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data
