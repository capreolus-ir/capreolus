import json
import os
import re
from collections import defaultdict, Counter
from os.path import join, exists

import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.extractor.common import get_profile_term_weight_user, get_profile_term_weight_topic, \
    get_domain_specific_term_weights, get_file_name
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
        "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
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
        maxqlen = 50
        maxdoclen = 5000

        query_cut = None
        document_cut = None
        alldomains = "travel,food,book"  # TODO this could be moved to benchmark, when rebased...

        if query_cut is not None and query_cut not in ["most_frequent", "topic-alltopics", "topic-amazon", "user-allusers", "user-amazon",
                                                       "unique_most_frequent", "unique_topic-alltopics", "unique_topic-amazon", "unique_user-allusers", "unique_user-amazon"]:
            raise ValueError(f"Value for query_cut is wrong {query_cut}")

        if document_cut is not None and document_cut not in ["most_frequent", "all_domains_tf", "all_domains_df", "amazon_tf", "amazon_df",
                                                             "unique_most_frequent", "unique_all_domains_tf", "unique_all_domains_df", "unique_amazon_tf", "unique_amazon_df"]:
            raise ValueError(f"Value for document_cut is wrong {document_cut}")

# let's add 2 parameters: 1-query-cut 2-doc-cut to define the ways to cut the query and document. if None, it would just truncate them.
# book: #docs: 1231 maxlen: 80917 avglen: 4559.179528838343
# travel #docs: 352 maxlen: 26468 avglen: 4239.082386363636
# food:  #docs: 995 maxlen: 1396  avglen: 475.06532663316585
#movie: #docs:  886 maxlen: 1037  avglen: 298.0056433408578
    @property
    def query_vocab_specific(self):
        return self.cfg["query_cut"]

    @property
    def domain_vocab_specific(self):
        return self.cfg["document_cut"]

    @property
    def all_domains(self):
        return self.cfg["alldomains"].split(",")

    def _get_pretrained_emb(self):
        magnitude_cache = CACHE_BASE_PATH / "magnitude/"
        return Magnitude(MagnitudeUtils.download_model(self.embed_paths[self.cfg["embeddings"]], download_dir=magnitude_cache))

    def _build_vocab(self, qids, docids, topics, querytype=None):
        tokenize = self["tokenizer"].tokenize
        self.qid2toks = {qid: tokenize(topics[qid].replace("]", "").replace("[", "")) for qid in qids}  # removing the entity mention tags
        if self.query_vocab_specific is not None:
            self.build_sorted_query_terms(qids, querytype)
        self.docid2toks = {docid: tokenize(self["index"].get_doc(docid)) for docid in docids}
        if self.domain_vocab_specific is not None:
            self.build_sorted_document_terms(docids)
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

    def create(self, qids, docids, topics, qdocs=None, querytype=None):

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

        logger.debug("build vocab")
        self._build_vocab(qids, docids, topics, querytype)
        self._build_embedding_matrix()

    def build_sorted_query_terms(self, qids, querytype):
        if self.query_vocab_specific in ["unique_most_frequent", "most_frequent"]:
            for qid in qids:
                terms = self.qid2toks[qid]
                term_counts = Counter(terms)
                if self.query_vocab_specific.startswith("unique"):
                    self.qid2toks[qid] = [t for t, v in sorted(term_counts.items(), key=lambda item:item[1], reverse=True)]
                else:
                    self.qid2toks[qid] = []
                    for t, v in sorted(term_counts.items(), key=lambda item: item[1], reverse=True):
                        self.qid2toks[qid].extend(list(map(str, np.repeat(t, v))))

        elif self.query_vocab_specific.startswith("unique_user") or self.query_vocab_specific.startswith("user"):
            user_term_weights = get_profile_term_weight_user(qids, querytype, self.cfg['query_cut'], self.query_vocab_specific[self.query_vocab_specific.index("-"):], self["tokenizer"])
            if self.query_vocab_specific.startswith("unique_user"):
                for qid in qids:
                    uid = qid.split("_")[-1]
                    sorted_terms = [t for t, v in sorted(user_term_weights[uid].items(), key=lambda item: item[1], reverse=True) if t in self.qid2toks[qid]]
                    self.qid2toks[qid] = sorted_terms
            else:
                for qid in qids:
                    uid = qid.split("_")[-1]
                    sorted_terms = []
                    terms = self.qid2toks[qid]
                    term_counts = Counter(terms)
                    for t, v in sorted(user_term_weights[uid].items(), key=lambda item: item[1], reverse=True):
                        if t in self.qid2toks[qid]:
                            sorted_terms.extend(list(map(str, np.repeat(t, term_counts[t]))))
                    self.qid2toks[qid] = sorted_terms

        elif self.query_vocab_specific.startswith("unique_topic") or self.query_vocab_specific.startswith("topic"):
            term_weights = get_profile_term_weight_topic(qids, querytype, self.cfg['query_cut'], self.query_vocab_specific[self.query_vocab_specific.index("-"):], self["tokenizer"])
            sorted_weights = sorted(term_weights.items(), key=lambda item: item[1], reverse=True)
            if self.query_vocab_specific.startswith("unique_topic"):
                for qid in qids:
                    sorted_terms = [t for t, v in sorted_weights if t in self.qid2toks[qid]]
                    self.qid2toks[qid] = sorted_terms
            else:
                for qid in qids:
                    sorted_terms = []
                    terms = self.qid2toks[qid]
                    term_counts = Counter(terms)
                    for t, v in sorted_weights:
                        if t in self.qid2toks[qid]:
                            sorted_terms.extend(list(map(str, np.repeat(t, term_counts[t]))))
                    self.qid2toks[qid] = sorted_terms

    def build_sorted_document_terms(self, docids):
        if self.domain_vocab_specific in ["unique_most_frequent", "most_frequent"]:
            for docid in docids:
                terms = self.docid2toks[docid]
                term_counts = Counter(terms)
                if self.domain_vocab_specific.startswith("unique"):
                    self.docid2toks[docid] = [t for t, v in sorted(term_counts.items(), key=lambda item:item[1], reverse=True)]
                else:
                    self.docid2toks[docid] = []
                    for t, v in sorted(term_counts.items(), key=lambda item: item[1], reverse=True):
                        self.docid2toks[docid].extend(list(map(str, np.repeat(t, v))))
            return
        elif self.domain_vocab_specific in ['all_domains_tf', 'unique_all_domains_tf']:
            term_weights = get_domain_specific_term_weights("all_domains", "tf", docids,
                                                            self.all_domains, self["index"], self["tokenizer"])
        elif self.domain_vocab_specific in ['all_domains_df', 'unique_all_domains_df']:
            term_weights = get_domain_specific_term_weights("all_domains", "df", docids,
                                                            self.all_domains, self["index"], self["tokenizer"])
        elif self.domain_vocab_specific in ['amazon_tf', 'unique_amazon_tf']:
            term_weights = get_domain_specific_term_weights("amazon", "tf", docids,
                                                            self.all_domains, self["index"], self["tokenizer"])
        elif self.domain_vocab_specific in ['amazon_df', 'unique_amazon_df']:
            term_weights = get_domain_specific_term_weights("amazon", "df", docids,
                                                            self.all_domains, self["index"], self["tokenizer"])
        else:
            raise RuntimeError(f"did not load term_weights {self.cfg['document_cut']}")

        sorted_weights = sorted(term_weights.items(), key=lambda item: item[1], reverse=True)
        if self.domain_vocab_specific.startswith("unique"):
            for docid in docids:
                sorted_terms = [t for t, v in sorted_weights if t in self.docid2toks[docid]]
                self.docid2toks[docid] = sorted_terms
        else:
            for docid in docids:
                sorted_terms = []
                terms = self.docid2toks[docid]
                term_counts = Counter(terms)
                for t, v in sorted_weights:
                    if t in self.docid2toks[docid]:
                        sorted_terms.extend(list(map(str, np.repeat(t, term_counts[t]))))
                self.docid2toks[docid] = sorted_terms

    def _tok2vec(self, toks):
        # return [self.embeddings[self.stoi[tok]] for tok in toks]
        return [self.stoi[tok] for tok in toks]

    def id2vec(self, qid, posid, negid=None, query=None):
        if query is not None:
            raise RuntimeError("we did not implement for query is not None yet")
            # if qid is None:
            #     query = self["tokenizer"].tokenize(query) # todo: I think here I should sort them based on what I want also(bur this is probably not used)
            #     pass
            # else:
            #     raise RuntimeError("received both a qid and query, but only one can be passed")

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

###TODO: to release the code, first clean all the additional and not neseccary caching (for debugging and viewing) which makes the parallel running a manual job!!!
class DocStats(Extractor):
    name = "docstats"
    dependencies = {
        "index": Dependency(module="index", name="anserini", config_overrides={"indexstops": True, "stemmer": "none"}),
        "backgroundindex": Dependency(module="index", name="anserinicorpus", config_overrides={"indexcorpus": "anserini0.9-index.clueweb09.englishonly.nostem.stopwording"}),##the other one could be:anserini0.9-index.clueweb09.englishonly.porterstem.stopwording
        "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
#        "tokenizerquery": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False, 'removesmallerlen': 2}), #only in PES20 paper: removesmallerlen is actually only used for user profile (not the short queries) but I cannot separate them
       # "tokenizer": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False}), # in PES20 paper I used spacy
        "entitylinking": Dependency(module="entitylinking", name='ambiversenlu'),
        "domainrelatedness": Dependency(module='entitydomainrelatedness', name='wiki2vecrepresentative',),
        "entityspecificity": Dependency(module='entityspecificity', name='higherneighborhoodmean',),
#       "entityspecificity": Dependency(module='entityspecificity', name='twohoppath'),
    }
    #TODO: maybe make one dependency like entity-handling? and then move these into that one.? maybe not.

    @staticmethod
    def config():
        entity_strategy = None
        query_vocab_specific = None # this is profile term weighting (on profiles)
        domain_vocab_specific = None # this is domain term weighting (on docs)
        onlyNamedEntities = False
        alldomains = "travel,food,book" # TODO this could be moved to benchmark, when rebased...

        if entity_strategy not in [None, 'all', 'domain', 'specific_domainrel']:  # TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        if query_vocab_specific is not None and not re.match(r"^(topic-alltopics|topic-amazon|user-allusers|user-amazon)_tf_k(\d+|-1)$", query_vocab_specific):
            raise ValueError(f"invalid query_vocab_specific: {query_vocab_specific}")

        # cutting for this and fq, would be setting the rest of the weights = 0, as we multiply them by termscore
        # k-1 means that we are reweighting and not cutting them! TODO Add other G corpuses
        if domain_vocab_specific is not None and not re.match(r"^(all_domains|amazon)_(tf|df)_k(\d+|-1)$", domain_vocab_specific):
            raise ValueError(f"invalid domain vocab specific {domain_vocab_specific}")

    @property
    def entity_strategy(self):
        return self.cfg["entity_strategy"]

    @property
    def query_vocab_specific(self):
        return self.cfg["query_vocab_specific"]

    @property
    def domain_vocab_specific(self):
        return self.cfg["domain_vocab_specific"]

    @property
    def all_domains(self):
        return self.cfg["alldomains"].split(",")

    def exist(self):
        return hasattr(self, "doc_tf")

    def get_profile_term_prob_cache_path(self):
        return self.get_cache_path() / 'profiletermprobs'

    def get_domain_term_weight_cache_file(self):
        return self.get_cache_path() / "domaintermweight.json"

    def get_selected_entities_cache_path(self):
        # logger.debug(self.get_cache_path() / 'selectedentities')
        return self.get_cache_path() / 'selectedentities'

    def create(self, qids, docids, topics, qdocs=None, querytype=None):  # TODO make changes to not use benchmark querytype anymore
        logger.debug(f"cache path: {self.get_cache_path()}")
        # todo remove these, just for initial checks
        logger.debug(qids)
        logger.debug(docids)
        # logger.debug(topics)
        # Todo where can I check this: is here good?
        if "nostem" in self["backgroundindex"].cfg["indexcorpus"]:
            if 'stemmer' in self["tokenizer"].cfg and self["tokenizer"].cfg['stemmer'] != "none":
                print("WARNING: tokenizer's stemming is on, but backgroundindex is without stemming.")
        else:
            if 'stemmer' not in self["tokenizer"].cfg or self["tokenizer"].cfg['stemmer'] == "none":
                print("WARNING: tokenizer's stemming is off, but backgroundindex is stemmed.")

        if self.exist():
            return

        logger.debug("Creating index")
        self["index"].create_index()
        logger.debug("Opening background index")
        self["backgroundindex"].open()

        if self.entity_strategy is not None:
            logger.debug("extracting entities from queries(user profiles)")
            for qid in qids:
                # To avoid redundancy in extracting (and as the user profiles are the same as many queries). We cache the extraction based on the profile_id/query_id.
                # This is handled in entitylinking component. In case of using another benchmark there may be a need to extend.
                self["entitylinking"].extract_entities(qid, topics[qid])

            logger.debug("loading entity descriptions")
            self["entitylinking"].load_descriptions()

        if self.entity_strategy == 'domain':
            self["domainrelatedness"].initialize(self["entitylinking"].get_cache_path())
        elif self.entity_strategy == 'specific_domainrel':
            self["domainrelatedness"].initialize(self["entitylinking"].get_cache_path())
            self["entityspecificity"].initialize()

        logger.debug("tokenizing queries [+entity descriptions]")
        # if logger.level in [logging.DEBUG]:
        if not exists(self.get_profile_term_prob_cache_path()):
            os.makedirs(self.get_profile_term_prob_cache_path(), exist_ok=True)
        if not exists(self.get_selected_entities_cache_path()):
            os.makedirs(self.get_selected_entities_cache_path(), exist_ok=True)

        self.qid_term_frequencies = {}
        self.qid_termprob = {}
        for qid in qids:
            qtext = topics[qid]
            qtext = qtext.replace("[", "")
            qtext = qtext.replace("]", "")

            qdesc = []
            qentities = self.get_entities(qid)  # {"NE": [...], "C": [...]}
            # since I just wanted to use this as a debug step, I didn't read from it when it was available
            # if logger.level in [logging.DEBUG]:
            entoutf = join(self.get_selected_entities_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
            if not exists(entoutf):
                with open(entoutf, 'w') as f:
                    f.write(json.dumps(qentities, indent=4))

#            logger.debug(f"{self.entity_strategy}: {qentities}")
#            logger.debug(f"qid: {qid} - {qentities}")
            for e in qentities["NE"]:
                qdesc.append(self["entitylinking"].get_entity_description(e))
            for e in qentities["C"]:
                qdesc.append(self["entitylinking"].get_entity_description(e))

            qtext += "\n" + "\n".join(qdesc)
            qtext = qtext.strip()
            query = self["tokenizer"].tokenize(qtext)

            q_count = Counter(query)
            self.qid_term_frequencies[qid] = {k: v for k, v in q_count.items()}
            self.qid_termprob[qid] = {k: (v/len(query)) for k, v in q_count.items()}

        # Here we calculate profile-term-weights based on the profile_topic or profile_user
        # Later we cut based on these weights or multiply the weight by the term-score (we are doing the latter now)
        # cutting could be as easy as setting other weights to zero.
        if self.query_vocab_specific is not None:
            logger.debug("creating profile term weights")
            m = re.match(r"^(topic|user)-(alltopics|amazon|allusers)_tf_k(\d+|-1)$", self.query_vocab_specific)
            if m:
                filter_by = m.group(1)
                filter_by_corpus = m.group(2)
                filter_topk = int(m.group(3))
                if filter_by == 'topic' and filter_by_corpus == 'allusers':
                    raise ValueError(f"invalid query_vocab_specific: {self.query_vocab_specific}")
                if filter_by == 'user' and filter_by_corpus not in ['allusers', 'amazon']:
                    raise ValueError(f"invalid query_vocab_specific: {self.query_vocab_specific}")

                # later used in rerankers, to multiply the term weight by the weights calculated in profile_term_weight
                self.profile_term_weight_by = filter_by
                self.profile_term_weight_by_corpus = filter_by_corpus

                if self.profile_term_weight_by == 'topic':
                    self.profile_term_weight = get_profile_term_weight_topic(qids, self["entitylinking"].get_benchmark_querytype(), self.profile_term_weight_by, self.profile_term_weight_by_corpus,
                                                                             self["tokenizer"], True, self.get_selected_entities_cache_path(), self["entitylinking"]) #term -> weight
                elif self.profile_term_weight_by == 'user':
                    self.profile_term_weight = get_profile_term_weight_user(qids, self["entitylinking"].get_benchmark_querytype(), self.profile_term_weight_by, self.profile_term_weight_by_corpus,
                                                                            self["tokenizer"], True, self.get_selected_entities_cache_path(), self["entitylinking"]) #uid -> term -> weight

        # since I just wanted to use this as a debug step, I didn't read from it when it was available
        for qid in qids:
            # if logger.level in [logging.DEBUG]:
            tfoutf = join(self.get_profile_term_prob_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
            if not exists(tfoutf):
                with open(tfoutf, 'w') as f:
                    sortedTP = {k: v for k, v in sorted(self.qid_termprob[qid].items(), key=lambda item: item[1], reverse=True)}
                    f.write(json.dumps(sortedTP, indent=4))

        logger.debug("tokenizing documents")
        self.doc_tf = {}
        self.doc_len = {}
        for docid in docids:
            # TODO is anserini's tokenizer removing the same punctuation as spacy was?
            # todo spacy tokenizer is added with some parameters... some of them should be cleaned  (see more in the tokenizer class) but it is very slow
            doc = self["tokenizer"].tokenize(self["index"].get_doc(docid))
            self.doc_tf[docid] = Counter(doc)
            self.doc_len[docid] = len(doc)

        # Here we calculate domain-vocab-term-specificity weights.
        # Then these weights are used in the rerankers.
        # we do this by multiplyinh these weights by the term-score of each reranker. TODO But other things also could be done!
        if self.domain_vocab_specific is not None:
            logger.debug("creating domain term weights")
            m = re.match(r"^(all_domains|amazon)_(tf|df)_k(\d+|-1)$", self.domain_vocab_specific)
            if m:
                domain_vocab_sp_general_corpus = m.group(1)
                domain_vocab_sp_tf_or_df = m.group(2)
                domain_vocab_sp_cut_at_k = int(m.group(3))
                if domain_vocab_sp_cut_at_k != -1:
                    raise ValueError(f"domain_vocab_sp_cut_at_k is not implemented!")

            self.domain_term_weight = get_domain_specific_term_weights(domain_vocab_sp_general_corpus, domain_vocab_sp_tf_or_df, docids,
                                                                       self.all_domains, self["index"], self["tokenizer"])
            tfoutf = self.get_domain_term_weight_cache_file()
            if not exists(tfoutf):
                with open(tfoutf, 'w') as f:
                    sortedweights = {k: v for k, v in sorted(self.domain_term_weight.items(), key=lambda item: item[1], reverse=True)}
                    f.write(json.dumps(sortedweights, indent=4))


        # todo: I have removed "Fixed partner" and "(or with a ..)" from the profiles [locally] (this should be done in the data on the servers and before publishing the data)

        #TODO: we have to calculate the avg doc len of the given query and documents eventually here (that's why O need qdocs as input) and here is the code but I disabled it for test:

        logger.debug("calculating average document length")
        self.query_avg_doc_len = {}
        for qid, docs in qdocs.items():
            doclen = 0
            for docid in docs:
                doclen += self.doc_len[docid]
            self.query_avg_doc_len[qid] = doclen/len(docs)
        logger.debug("extractor DONE")

    def background_idf(self, term):# TODO could be replaced by that: the index itself has a function for idf, but it has a +1...
        df = self["backgroundindex"].get_df(term)
        total_docs = self["backgroundindex"].numdocs
        return np.log10((total_docs - df + 0.5) / (df + 0.5))

    def background_termprob(self, term):
        tf = self["backgroundindex"].get_tf(term)
        total_terms = self["backgroundindex"].numterms
        return tf/ total_terms

    def get_entities(self, profile_id):
        if self.entity_strategy is None:
            return {"NE": [], "C": []} #TODO propagate this change to what calls this function
        elif self.entity_strategy == 'all':
            ret = self['entitylinking'].get_all_entities(profile_id)
        elif self.entity_strategy == 'domain':
            ret = self["domainrelatedness"].get_domain_related_entities(
                profile_id, self['entitylinking'].get_all_entities(profile_id)
            )
        elif self.entity_strategy == 'specific_domainrel':
            ret = self['entityspecificity'].top_specific_entities(
                profile_id, self["domainrelatedness"].get_domain_related_entities(
                    profile_id, self['entitylinking'].get_all_entities(profile_id)
                )
            )
        else:
            raise NotImplementedError("TODO implement other entity strategies (by first implementing measures)")

        if self.cfg["onlyNamedEntities"]:
            return {"NE": ret["NE"], "C": []}

        return ret

    def id2vec(self, qid, posid, negid=None, query=None):
        # if query is not None:
        #     # if qid is None:
        #     #     query = self["tokenizer"].tokenize(query)
        #     # else:
        #     #     raise RuntimeError("received both a qid and query, but only one can be passed")
        #     raise RuntimeError("this is not implemented completely to get the query")
        # else:
        #     query = self.qid_termprob[qid]
        if qid is None:
            raise RuntimeError("this is not implemented completely to get the query")
        return {"qid": qid, "posdocid": posid} #these are what used in BM25 and LM rankers, we could give other things here, but we are just getting them from extractor
#if you want to implement cutting docs or queries, you could actually do it here but need to change the BM25 and LM rerankers as well!


class DocStatsEmbedding(DocStats):
    name = "docstatsembedding"

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
        entity_strategy = None
        query_vocab_specific = None # this is profile term weighting (on profiles)
        domain_vocab_specific = None # this is domain term weighting (on docs)
        onlyNamedEntities = False
        alldomains = "travel,food,book"  # TODO this could be moved to benchmark, when rebased...

        if entity_strategy not in [None, 'all', 'domain', 'specific_domainrel']:  # TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        if query_vocab_specific is not None and not re.match(r"^(topic-alltopics|topic-amazon|user-allusers|user-amazon)_tf_k(\d+|-1)$", query_vocab_specific):
            raise ValueError(f"invalid query_vocab_specific: {query_vocab_specific}")


        # k-1 means that we are reweighting and not cutting them! TODO Add other G corpuses
        if domain_vocab_specific is not None and not re.match(r"^(all_domains|amazon)_(tf|df)_k(\d+|-1)$", domain_vocab_specific):
            raise ValueError(f"invalid domain vocab specific {domain_vocab_specific}")

        embeddings = "w2vnews"

    def _get_pretrained_emb(self):
        gensim_cache = CACHE_BASE_PATH / "gensim/"
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache.absolute())

        import gensim
        import gensim.downloader as api

        model_path = api.load(self.embed_names[self.cfg["embeddings"]], return_path=True)
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    def create(self, qids, docids, topics, qdocs=None, querytype=None):
        if self.exist():
            return

        super().create(qids, docids, topics, qdocs, querytype=None)

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

    def id2vec(self, qid, posid, negid=None, query=None):
        # if query is not None:
        #     # if qid is None:
        #     #     query = self["tokenizer"].tokenize(query)
        #     # else:
        #     #     raise RuntimeError("received both a qid and query, but only one can be passed")
        #     raise RuntimeError("this is not implemented completely to get the query")
        # else:
        #     query = self.qid_termprob[qid]
        if qid is None:
            raise RuntimeError("this is not implemented completely to get the query")
        return {"qid": qid, "posdocid": posid} #these are what used in BM25 and LM rankers, we could give other things here, but we are just getting them from extractor
