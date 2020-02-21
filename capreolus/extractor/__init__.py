from collections import defaultdict

import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.registry import ModuleBase, RegisterableModule, Dependency
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
                    self.idf[tok] = self["index"].getidf(tok)

        logger.debug(f"added {len(self.stoi)-n_words_before} terms to the stoi of extractor {self.name}")


class EmbedText(Extractor):
    name = "embedtext"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"keepstops": True}),
        "tokenizer": Dependency(module="tokenizer", name="anserini"),
    }

    pad = 0
    pad_tok = "<pad>"
    embed_pathes = {
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
        maxdoclen = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.qid2toks = defaultdict(list)
        self.docid2toks = defaultdict(list)
        self.idf = defaultdict(lambda: 0)
        self.embeddings = None
        # self.cache = self.load_cache()    # TODO

    def _get_pretrained_emb(self):
        return Magnitude(
            MagnitudeUtils.download_model(self.embed_pathes[self.cfg["embeddings"]], download_dir=self.get_cache_path())
        )

    def _build_vocab(self, qids, docids, topics):
        tokenize = self["tokenizer"].tokenize
        self.qid2toks = {qid: tokenize(topics.get(qid, "")) for qid in qids}
        self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
        self._extend_stoi(self.qid2toks.values(), calc_idf=self.cfg["calcidf"])
        self._extend_stoi(self.docid2toks.values(), calc_idf=self.cfg["calcidf"])
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.info(f"vocabulary constructued, with {len(self.itos)} terms in total")

    def _get_idf(self, toks):
        return [self.idf.get(tok, 0) for tok in toks]

    def _build_embedding_matrix(self):
        assert len(self.stoi) > 1  # needs more vocab than self.pad_tok

        magnitude_emb = self._get_pretrained_emb()
        emb_dim = magnitude_emb.dim
        embed_vocab = set(term for term, _ in magnitude_emb)
        embed_matrix = np.zeros((len(self.stoi), emb_dim), dtype=np.float32)

        n_missed = 0
        for term, idx in self.stoi.items():
            if term in embed_vocab:
                embed_matrix[idx] = magnitude_emb.query(term)
            elif term == self.pad_tok:
                embed_matrix[idx] = np.zeros(emb_dim)
            else:
                n_missed += 1
                embed_matrix[idx] = np.zeros(emb_dim) if self.cfg["zerounk"] else np.random.normal(scale=0.5, size=emb_dim)

        logger.info(f"Embedding matrix {self.cfg['embeddings']} constructued, with shape {embed_matrix.shape}")
        if n_missed > 0:
            logger.warning(f"{n_missed}/{len(self.stoi)} (%.3f) term missed" % (n_missed / len(self.stoi)))

        self.embeddings = embed_matrix

    def exist(self):
        return isinstance(self.embeddings, np.ndarray) and 0 < len(self.stoi) == self.embeddings.shape[0]

    def create(self, qids, docids, topics):

        if self.exist():
            return

        self["index"].create_index()

        self._build_vocab(qids, docids, topics)
        self._build_embedding_matrix()

    def id2vec(self, qid, posid, negid=None):
        def _tok2vec(toks):
            return [self.embeddings[self.stoi[tok]] for tok in toks]

        qlen, doclen = self.cfg["maxqlen"], self.cfg["maxdoclen"]
        query, posdoc = self.qid2toks.get(qid, None), self.docid2toks.get(posid, None)
        if not posdoc:
            raise MissingDocError(qid, posid)

        idfs = padlist(self._get_idf(query), qlen, 0)
        query = _tok2vec(padlist(query, qlen, self.pad_tok))
        posdoc = _tok2vec(padlist(posdoc, doclen, self.pad_tok))

        data = {
            "qid": qid,
            "posdocid": posid,
            "idfs": np.array(idfs, dtype=np.float32),
            "query": np.array(query, dtype=np.float32),
            "posdoc": np.array(posdoc, dtype=np.float32),
            "query_idf": np.array(idfs, dtype=np.float32),
        }

        if not negid:
            logger.debug(f"missing negtive doc id for qid {qid}")
            return data

        negdoc = self.docid2toks.get(negid, None)
        if not negdoc:
            raise MissingDocError(qid, negid)

        negdoc = _tok2vec(padlist(negdoc, doclen, self.pad_tok))
        data["negdocid"] = negid
        data["negdoc"] = np.array(negdoc, dtype=np.float32)

        return data
