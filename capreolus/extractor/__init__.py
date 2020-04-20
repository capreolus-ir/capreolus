import os
from collections import defaultdict, Counter

import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

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

    def _get_pretrained_emb(self):
        magnitude_cache = CACHE_BASE_PATH / "magnitude/"
        return Magnitude(MagnitudeUtils.download_model(self.embed_paths[self.cfg["embeddings"]], download_dir=magnitude_cache))

    def _build_vocab(self, qids, docids, topics):
        tokenize = self["tokenizer"].tokenize
        self.qid2toks = {qid: tokenize(topics[qid]) for qid in qids}
        self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
        self._extend_stoi(self.qid2toks.values(), calc_idf=self.cfg["calcidf"])
        self._extend_stoi(self.docid2toks.values(), calc_idf=self.cfg["calcidf"])
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.info(f"vocabulary constructed, with {len(self.itos)} terms in total")

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

    def id2vec(self, qid, posid, negid=None, query=None):
        if query is not None:
            if qid is None:
                query = self["tokenizer"].tokenize(query)
                pass
            else:
                raise RuntimeError("received both a qid and query, but only one can be passed")

        else:
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
        }

        if not negid:
            logger.debug(f"missing negtive doc id for qid {qid}")
            return data

        negdoc = self.docid2toks.get(negid, None)
        if not negdoc:
            raise MissingDocError(qid, negid)

        negdoc = self._tok2vec(padlist(negdoc, doclen, self.pad_tok))
        data["negdocid"] = negid
        data["negdoc"] = np.array(negdoc, dtype=np.long)

        return data


class DocStats(Extractor):
    name = "docstats"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
#        "tokenizerquery": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False, 'removesmallerlen': 2}), #removesmallerlen is actually only used for user profile (not the short queries) but I cannot separate them
       # "tokenizer": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False}),
    }

    @staticmethod
    def config():
        pass

    def exist(self):
        return hasattr(self, "doc_tf")

    def create(self, qids, docids, topics, qdocs=None):
        if self.exist():
            return

        self["index"].create_index()

        self.qid2toks = {}
        self.qid_termprob = {}
        for qid in qids:
            query = self["tokenizer"].tokenize(topics[qid])
            self.qid2toks[qid] = query
            q_count = Counter(query)
            self.qid_termprob[qid] = {k: (v/len(query)) for k, v in q_count.items()}

        # TODO hardcoded paths
        #df_fn, freq_fn = "/GW/NeuralIR/work/PES20/counts_IDF_stemmed.txt", "/GW/NeuralIR/work/PES20/counts_LM_stemmed.txt"
        #doclen_fn = "/GW/NeuralIR/work/PES20/counts_MUS_stemmed.txt"
        df_fn, freq_fn = "/GW/PKB/work/data_personalization/TREC_format/counts_IDF_stemmed_cw12.nostemming.txt", "/GW/PKB/work/data_personalization/TREC_format/counts_LM_stemmed_cw12.nostemming.txt"
        
        logger.debug("computing background probabilities")
        dfs = {}
        with open(df_fn, "rt") as f:
            for line in f:
                cidx = line.strip().rindex(",")
                k = line.strip()[:cidx]
                v = line.strip()[cidx + 1:]
                dfs[k] = int(v)

        total_docs = dfs["total_docs"]
        del dfs["total_docs"]

        # TODO unsure if log base is correct? gh:Yes I used the same; unsure if the non-negative max(0, idf) formulation was used, gh: I also didn't
        get_idf = lambda x: np.log10((total_docs - dfs[x] + 0.5) / (dfs[x] + 0.5))
        self.background_idf = {term: get_idf(term) for term in dfs}

        tfs = {}
        with open(freq_fn, "rt") as f:
            for line in f:
                cidx = line.strip().rindex(",")
                k = line.strip()[:cidx]
                v = line.strip()[cidx + 1:]
                tfs[k] = int(v)

        total_terms = tfs["total_terms"]
        del tfs["total_terms"]
        self.background_termprob = {term: tfs[term]/total_terms for term in tfs}

        logger.debug("tokenizing documents")
        self.doc_tf = {}
        self.doc_len = {}
        for docid in docids:
            # TODO is anserini's tokenizer removing the same punctuation as spacy was?
            # todo spacy tokenizer is added with some parameters... some of them should be cleaned  (see more in the tokenizer class)
            doc = self["tokenizer"].tokenize(self["index"].get_doc(docid))
            self.doc_tf[docid] = Counter(doc)
            self.doc_len[docid] = len(doc)

        # todo: I have removed "Fixed partner" and "(or with a ..)" from the profiles [locally] (this should be done in the data on the servers and before publishing the data)

        #TODO: we have to calculate the avg doc len of the given query and documents eventually here (that's why O need qdocs as input) and here is the code but I disabled it for test:

        self.query_avg_doc_len = {}
        for qid, docs in qdocs.items():
            doclen = 0
            for docid in docs:
                doclen += self.doc_len[docid]
            self.query_avg_doc_len[qid] = doclen/len(docs)

        #self.query_avg_doc_len = {}
        #with open(doclen_fn, "rt") as f:
        #    for line in f:
        #        qid, avglen = line.strip().split(",")
        #        self.query_avg_doc_len[qid] = int(avglen)

        # todo (problem): the calculated avgdoclength and the one loaded are not matching: the reason is that that one is calculated from 100 docs, this one from 20 docs.
        # shared_items = {k: list([self.query_avg_doc_len[k], query_avg_doc_len[k]]) for k in self.query_avg_doc_len if k in query_avg_doc_len and self.query_avg_doc_len[k] == query_avg_doc_len[k]}
        # diff_items = {k: list([self.query_avg_doc_len[k], query_avg_doc_len[k]]) for k in self.query_avg_doc_len if k in query_avg_doc_len and self.query_avg_doc_len[k] != query_avg_doc_len[k]}
        # print(len(shared_items), shared_items)
        # print(len(diff_items), diff_items) ## there are very different

    def id2vec(self, qid, posid, negid=None, query=None):#todo (ask) where is it used?
        if query is not None:
            if qid is None:
                query = self["tokenizer"].tokenize(query)
            else:
                raise RuntimeError("received both a qid and query, but only one can be passed")
        else:
            query = self.qid2toks[qid]

        return {"qid": qid, "posdocid": posid, "negdocid": negid}


class DocStatsEmbedding(DocStats):
    name = "docstatsembedding"
    dependencies = {# TODO is this okay like this? if this is changed here, would the parent functions also use differently??
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        # "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
        "tokenizerquery": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False, 'removesmallerlen': 2}), #removesmallerlen is actually only used for user profile (not the short queries) but I cannot separate them
        "tokenizer": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False}),
    }

    embed_names = {
        "glove6b": "glove-wiki-gigaword-300",
        "glove6b.50d": "glove-wiki-gigaword-50",
        "w2vnews" : "word2vec-google-news-300",
        "fasttext": "fasttext-wiki-news-subwords-300", #TODO: "Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)" is this model the same as one used with Magnitude?

    }

    def exist(self):
        return hasattr(self, "similarity_matrix")

    @staticmethod
    def config():
        embeddings = "w2vnews"

    def _get_pretrained_emb(self):
        gensim_cache = CACHE_BASE_PATH / "gensim/"
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache.absolute())

        import gensim
        import gensim.downloader as api

        model_path = api.load(self.embed_names[self.cfg["embeddings"]], return_path=True)
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    def create(self, qids, docids, topics, qdocs=None):
        if self.exist():
            return

        super().create(qids, docids, topics, qdocs)

        logger.debug("loading embedding")
        self.emb_model = self._get_pretrained_emb()


    def get_term_occurrence_probability(self, qterm, docterm, docid, threshold):
        nu = self.emb_model.similarity(qterm, docterm) if (self.emb_model.__contains__(qterm) and self.emb_model.__contains__(docterm)) else 0
        if nu < threshold:
            return 0

        de = 0
        for term in self.doc_tf[docid].keys():#TODO could I precalc this? There is the threshold which is the reranker parameter... Should I make it the extractor parameter??
            temp_sim = self.emb_model.similarity(term, docterm) if (self.emb_model.__contains__(term) and self.emb_model.__contains__(docterm)) else 0
            de += temp_sim if temp_sim >= threshold else 0

        return nu/de

    def id2vec(self, qid, posid, negid=None, query=None):#todo change this later ...
        if query is not None:
            if qid is None:
                query = self["tokenizer"].tokenize(query)
            else:
                raise RuntimeError("received both a qid and query, but only one can be passed")
        else:
            query = self.qid2toks[qid]

        return {"qid": qid, "posdocid": posid, "negdocid": negid}
