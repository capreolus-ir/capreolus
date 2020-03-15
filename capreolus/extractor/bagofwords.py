from capreolus.extractor import Extractor
from capreolus.tokenizer import Tokenizer
from capreolus.utils.loginit import get_logger
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict

from capreolus.registry import Dependency

logger = get_logger(__name__)  # pylint: disable=invalid-name


class BagOfWords(Extractor):
    """ Bag of Words (or bag of trigrams when `datamode=trigram`) extractor. Used with the DSSM reranker. """

    name = "bagofwords"
    dependencies = {
        "index": Dependency(
            module="index",
            name="anserini",
            config_overrides={"indexstops": True, "stemmer": "none"},
        ),
        "tokenizer": Dependency(module="tokenizer", name="anserini"),
    }
    pad = 0
    pad_tok = "<pad>"
    tokenizer_name = "anserini"

    @staticmethod
    def config():
        datamode = "unigram"  # type of input: 'unigram' or 'trigram'
        keepstops = False  # include stopwords in the reranker's input
        maxqlen = 4
        maxdoclen = 800

    def get_trigrams_for_toks(self, toks_list):
        return [("#%s#" % tok)[i : i + 3] for tok in toks_list for i in range(len(tok))]

    def _build_vocab_unigram(self, qids, docids, topics):
        tokenize = self["tokenizer"].tokenize
        self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
        self.docid2toks = {
            docid: tokenize(self["index"].get_doc(docid)) for docid in docids
        }
        self._extend_stoi(self.qid2toks.values())
        self._extend_stoi(self.docid2toks.values())
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")

    def _build_vocab_trigram(self, qids, docids, topics):
        tokenize = self["tokenizer"].tokenize
        self.qid2toks = {
            qid: self.get_trigrams_for_toks(tokenize(topics[qid])) for qid in qids
        }
        self.docid2toks = {
            docid: self.get_trigrams_for_toks(tokenize(self["index"].get_doc(docid)))
            for docid in docids
        }
        self._extend_stoi(self.qid2toks.values())
        self._extend_stoi(self.docid2toks.values())
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")

    def _build_vocab(self, qids, docids, topics):
        if self.cfg["datamode"] == "unigram":
            self._build_vocab_unigram(qids, docids, topics)
        elif self.cfg["datamode"] == "trigram":
            self._build_vocab_trigram(qids, docids, topics)
        else:
            raise NotImplementedError
        self.embeddings = self.stoi

    def exist(self):
        return (
            hasattr(self, "qid2toks")
            and hasattr(self, "docid2toks")
            and len(self.stoi) > 1
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

    def id2vec(self, q_id, posdoc_id, negdoc_id=None, query=None):
        # TODO: Get rid of this if check. Standardize the interface
        if query is not None:
            if q_id is None:
                query_toks = self["tokenizer"].tokenize(query)
                pass
            else:
                raise RuntimeError(
                    "received both a qid and query, but only one can be passed"
                )

        else:
            query_toks = self.qid2toks[q_id]
        posdoc_toks = self.docid2toks.get(posdoc_id)

        if not posdoc_toks:
            logger.debug("missing docid %s", posdoc_id)
            return None

        transformed_query = self.transform_txt(query_toks, self.cfg["maxqlen"])
        idfs = [self.idf[self.itos[tok]] for tok, count in enumerate(transformed_query)]
        transformed = {
            "qid": q_id,
            "posdocid": posdoc_id,
            "query": transformed_query,
            "posdoc": self.transform_txt(posdoc_toks, self.cfg["maxdoclen"]),
            "query_idf": np.array(idfs, dtype=np.float32),
        }

        if negdoc_id is not None:
            negdoc_toks = self.docid2toks.get(negdoc_id)
            if not negdoc_toks:
                logger.debug("missing docid %s", negdoc_id)
                return None
            transformed["negdocid"] = negdoc_id
            transformed["negdoc"] = self.transform_txt(
                negdoc_toks, self.cfg["maxdoclen"]
            )

        return transformed

    def transform_txt(self, term_list, maxlen):
        nvocab = len(self.stoi)
        bog_txt = np.zeros(nvocab, dtype=np.float32)

        if self.cfg["datamode"] == "unigram":
            toks = [self.stoi.get(term, 0) for term in term_list]
            tok_counts = Counter(toks)
        elif self.cfg["datamode"] == "trigram":
            trigrams = self.get_trigrams_for_toks(term_list)
            toks = [self.stoi.get(trigram, 0) for trigram in trigrams]
            tok_counts = Counter(toks)
        else:
            raise Exception("Unknown datamode")

        for tok, count in tok_counts.items():
            bog_txt[tok] = count

        return bog_txt
