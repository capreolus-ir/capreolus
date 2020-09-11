import json
import logging
import os
import pickle
import re
from collections import defaultdict, Counter
from os import listdir
from os.path import join, exists

import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, CACHE_BASE_PATH, PACKAGE_PATH
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import padlist, get_file_name, get_user_profiles
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

    def create(self, qids, docids, topics, qdocs=None):

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
        "backgroundindex": Dependency(module="index", name="anserinicorpus", config_overrides={"indexcorpus": "anserini0.9-index.clueweb09.englishonly.nostem.stopwording"}),##the other one could be:anserini0.9-index.clueweb09.englishonly.porterstem.stopwording
        "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
#        "tokenizerquery": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False, 'removesmallerlen': 2}), #removesmallerlen is actually only used for user profile (not the short queries) but I cannot separate them
       # "tokenizer": Dependency(module="tokenizer", name="spacy", config_overrides={"keepstops": False}),
        "entitylinking": Dependency(module="entitylinking", name='ambiversenlu'),
        "domainrelatedness": Dependency(module='entitydomainrelatedness', name='wiki2vecrepresentative',),
        "entityspecificity": Dependency(module='entityspecificity', name='higherneighborhoodmean',),
#       "entityspecificity": Dependency(module='entityspecificity', name='twohoppath'),
    }

    @staticmethod
    def config():
        entity_strategy = None
        filter_query = None # this is profile term weighting (on profiles)
        domain_vocab_specific = None # this is domain term weighting (on docs)
        onlyNamedEntities = False

        if entity_strategy not in [None, 'all', 'domain', 'specific_domainrel']:  # TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        if filter_query is not None and not re.match(r"^(topic|user|amazon)_tf_k(\d+|-1)$", filter_query):
            raise ValueError(f"invalid filter query: {filter_query}")

        # k-1 means that we are reweighting and not cutting them! TODO Add other G corpuses
        if domain_vocab_specific is not None and not re.match(r"^(all_domains|amazon)_(tf|df)_k(\d+|-1)$", domain_vocab_specific):
            raise ValueError(f"invalid domain vocab specific {domain_vocab_specific}")

    @property
    def entity_strategy(self):
        return self.cfg["entity_strategy"]

    @property
    def filter_query(self):
        return self.cfg["filter_query"]

    @property
    def domain_vocab_specific(self):
        return self.cfg["domain_vocab_specific"]

    def exist(self):
        return hasattr(self, "doc_tf")

    def get_profile_term_prob_cache_path(self):
        return self.get_cache_path() / 'profiletermprobs'

    def get_domain_term_weight_cache_file(self):
        return self.get_cache_path() / "domaintermweight.json"

    def get_selected_entities_cache_path(self):
        # logger.debug(self.get_cache_path() / 'selectedentities')
        return self.get_cache_path() / 'selectedentities'

    def create(self, qids, docids, topics, qdocs=None):
        logger.debug(f"cache path: {self.get_cache_path()}")
        #todo where can I check this: is here good?
        if "nostem" in self["backgroundindex"].cfg["indexcorpus"]:
            if 'stemmer' in self["tokenizer"].cfg and self["tokenizer"].cfg['stemmer'] != "none":
                print("WARNING: tokenizer's stemming is on, but backgroundindex is without stemming.")
        else:
            if 'stemmer' not in self["tokenizer"].cfg or self["tokenizer"].cfg['stemmer'] == "none":
                print("WARNING: tokenizer's stemming is off, but backgroundindex is stemmed.")

        if self.exist():
            return

        self["index"].create_index()
        logger.debug("Openning background index")
        self["backgroundindex"].open()

        if self.entity_strategy is not None:
            logger.debug("extracting entities from queries(user profiles)")
            for qid in qids:
                # To avoid redundency in extracting (and as the user profiles are the same as many queries). We cache the extraction based on the profileid.
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
        if logger.level in [logging.DEBUG, logging.NOTSET]:
            os.makedirs(self.get_profile_term_prob_cache_path(), exist_ok=True)
        os.makedirs(self.get_selected_entities_cache_path(), exist_ok=True)

        # self.qid2toks = {}
        self.qidlen = {}
        self.qid_termprob = {}
        for qid in qids:
            qtext = topics[qid]
            qdesc = []
            
            entoutf = join(self.get_selected_entities_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
            if exists(entoutf):
                with open(entoutf, 'r') as f:
 #                   logger.debug(entoutf)
                    qentities = json.loads(f.read())
            else:
                qentities = self.get_entities(qid) # {"NE": [...], "C": [...]}
                with open(entoutf, 'w') as f:
                    f.write(json.dumps(qentities, indent=4))

#            logger.debug(f"{self.entity_strategy}: {qentities}")
#            logger.debug(f"qid: {qid} - {qentities}")
            for e in qentities["NE"]:
                qdesc.append(self["entitylinking"].get_entity_description(e))
            for e in qentities["C"]:
                qdesc.append(self["entitylinking"].get_entity_description(e))

            qtext += "\n" + "\n".join(qdesc)
            query = self["tokenizer"].tokenize(qtext)

            # self.qid2toks[qid] = query
            self.qidlen[qid] = len(query)
            q_count = Counter(query)
            self.qid_termprob[qid] = {k: (v/len(query)) for k, v in q_count.items()}

        # TODO re-implement this part carefully! it's not as easy as it sounds.
        #  user-specific is another thing... than using amazon or topic-specific.
        # # here we remove (keep-topk) or reweight the query terms:
        # # this does not work for inputs which is only the query (or coupled with it)
        # if self.filter_query is not None:
        #     m = re.match(r"^(topic|user|amazon)_tf_k(\d+|-1)$", self.filter_query)
        #     if m:
        #         filter_by = m.group(1)
        #         filter_topk = int(m.group(2)) #TODO implement
        #
        #     # let's change this to something easier... We don't mess with qid_termprob anymore!!!
        #     self.profile_vocab_specific = self.get_profile_specific_term_weights(filter_by)
        #
        #
        #     reweighted_qid_termprob = {}
        #     if filter_by == 'domain':
        #         baseprofiletype = profiletype.split("_")[0]
        #         domainprofiletype = profiletype[profiletype.find("_") + 1:]
        #
        #         # todo change: now it can only be used for KITT data so later change it to adapt to any
        #         benchmarkdir = "/GW/PKB/work/data_personalization/TREC_format/" # change these when rebasing to use the benchmark as inherited dependency
        #         userfullprofiles = get_user_profiles(join(benchmarkdir, f"book_topics.{baseprofiletype}.txt")) # book is an exemplary recommendation domain, and it does not matter which one is read since the profile is the same for all.
        #
        #         GD = self.get_domain_general_corpus(qids, baseprofiletype, userfullprofiles)
        #
        #         for qid in qids:
        #             uid = qid.split("_")[1]
        #             tfoutf = join(self.get_profile_term_prob_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
        #             if exists(tfoutf):
        #                 with open(tfoutf, 'r') as f:
        #                     self.qid_termprob[qid] = json.loads(f.read())
        #             else:
        #                 tfs = self.qid_termprob[qid]
        #                 reweighted_qid_termprob[qid] = {}
        #                 for v in tfs:
        #                     reweighted_qid_termprob[qid][v] = tfs[v] / GD[uid][v]
        #
        #                 # to get reweighted term frequencies (a probability distribution)
        #                 # we will divide every weight by the sum of the all of the weights.
        #                 sum_vals = sum(reweighted_qid_termprob[qid].values())
        #                 self.qid_termprob[qid] = {k: v/sum_vals for k, v in reweighted_qid_termprob[qid].items()}
        #
        #             # to get the query tokens (in another word word counts for query)
        #             # we cannot simply multiply this reweighted tf with the doc lenght
        #             # so we assume that the term with smallest reweighted tf occured once
        #             # and calculate counts based on that. Finally we use round to convert them to integers.
        #             min_reweighted_tf = min(self.qid_termprob[qid].values())
        #             # query_token_counts = {k: round(v / min_reweighted_tf) for k, v in self.qid_termprob[qid].items()}
        #             # self.qid2toks[qid] = []
        #             # for k, v in query_token_counts.items():
        #             #     self.qid2toks[qid] += np.repeat(k, v).tolist()
        #             self.qidlen[qid] = 1/min_reweighted_tf
        #
        #     elif filter_by == 'user':
        #         # create G (the accumulated other corpus of all users)
        #         GU = self.get_user_general_corpus(qids)
        #
        #         for qid in qids:
        #             tfoutf = join(self.get_profile_term_prob_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
        #             if exists(tfoutf):
        #                 with open(tfoutf, 'r') as f:
        #                     self.qid_termprob[qid] = json.loads(f.read())
        #             else:
        #                 tfs = self.qid_termprob[qid]
        #                 reweighted_qid_termprob[qid] = {}
        #                 for v in tfs:
        #                     reweighted_qid_termprob[qid][v] = tfs[v] / GU[v]
        #
        #                 # to get reweighted term frequencies (a probability distribution)
        #                 # we will divide every weight by the sum of the all of the weights.
        #                 sum_vals = sum(reweighted_qid_termprob[qid].values())
        #                 self.qid_termprob[qid] = {k: v/sum_vals for k, v in reweighted_qid_termprob[qid].items()}
        #
        #             # to get the query tokens (in another word word counts for query)
        #             # we cannot simply multiply this reweighted tf with the doc lenght
        #             # so we assume that the term with smallest reweighted tf occured once
        #             # and calculate counts based on that. Finally we use round to convert them to integers.
        #             min_reweighted_tf = min(self.qid_termprob[qid].values())
        #             # query_token_counts = {k: round(v / min_reweighted_tf) for k, v in self.qid_termprob[qid].items()}
        #             # self.qid2toks[qid] = []
        #             # for k, v in query_token_counts.items():
        #             #     self.qid2toks[qid] += np.repeat(k, v).tolist()
        #             self.qidlen[qid] = 1/min_reweighted_tf


        for qid in qids:
            if logger.level in [logging.DEBUG, logging.NOTSET]:  # since I just wanted to use this as a debug step, I didn't read from it when it was available
                tfoutf = join(self.get_profile_term_prob_cache_path(), get_file_name(qid, self["entitylinking"].get_benchmark_name(), self["entitylinking"].get_benchmark_querytype()))
                if not exists(tfoutf):
                    with open(tfoutf, 'w') as f:
                        sortedTP = {k: v for k, v in sorted(self.qid_termprob[qid].items(), key=lambda item: item[1], reverse=True)}
                        f.write(json.dumps(sortedTP, indent=4))

        # TODO hardcoded paths
        #df_fn, freq_fn = "/GW/NeuralIR/work/PES20/counts_IDF_stemmed.txt", "/GW/NeuralIR/work/PES20/counts_LM_stemmed.txt"
        #doclen_fn = "/GW/NeuralIR/work/PES20/counts_MUS_stemmed.txt"
        #df_fn, freq_fn = "/home/ghazaleh/workspace/capreolus/data/PES20/counts_IDF_stemmed.txt", "/home/ghazaleh/workspace/capreolus/data/PES20/counts_LM_stemmed.txt"
        #doclen_fn = "/home/ghazaleh/workspace/capreolus/data/PES20/counts_MUS_stemmed.txt"
        # df_fn, freq_fn = "/GW/PKB/work/data_personalization/TREC_format/counts_IDF_stemmed_cw12.nostemming.txt", "/GW/PKB/work/data_personalization/TREC_format/counts_LM_stemmed_cw12.nostemming.txt"
        
        # logger.debug("computing background probabilities")
        # dfs = {}
        # with open(df_fn, "rt") as f:
        #     for line in f:
        #         cidx = line.strip().rindex(",")
        #         k = line.strip()[:cidx]
        #         v = line.strip()[cidx + 1:]
        #         dfs[k] = int(v)
        #         # df_bg = self["backgroundindex"].get_df(k)
        #         # if v != df_bg:
        #         #     logger.debug("df do noe match: {}".format(k))
        #
        # total_docs = dfs["total_docs"]
        # del dfs["total_docs"]

        # TODO unsure if log base is correct? gh:Yes I used the same; unsure if the non-negative max(0, idf) formulation was used, gh: I also didn't
        # get_idf = lambda x: np.log10((total_docs - dfs[x] + 0.5) / (dfs[x] + 0.5))
        # self.background_idfs = {term: get_idf(term) for term in dfs}

        # tfs = {}
        # with open(freq_fn, "rt") as f:
        #     for line in f:
        #         cidx = line.strip().rindex(",")
        #         k = line.strip()[:cidx]
        #         v = line.strip()[cidx + 1:]
        #         tfs[k] = int(v)
        #
        # total_terms = tfs["total_terms"]
        # del tfs["total_terms"]
        # self.background_termprob = {term: tfs[term]/total_terms for term in tfs}

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
            tfoutf = self.get_domain_term_weight_cache_file()
            if exists(tfoutf):
                with open(tfoutf, 'r') as f:
                    self.domain_term_weight = json.loads(f.read())
            else:
                logger.debug("creating domain term weights")
                m = re.match(r"^(all_domains|amazon)_(tf|df)_k(\d+|-1)$", self.domain_vocab_specific)
                if m:
                    domain_vocab_sp_general_corpus = m.group(1)
                    domain_vocab_sp_tf_or_df = m.group(2)
                    domain_vocab_sp_cut_at_k = int(m.group(3))
                    if domain_vocab_sp_cut_at_k != -1:
                        raise ValueError(f"domain_vocab_sp_cut_at_k is not implemented!")
                self.domain_term_weight = self.get_domain_specific_term_weights(domain_vocab_sp_general_corpus, domain_vocab_sp_tf_or_df, docids)

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

    # TODO: implement
    # def get_profile_specific_term_weights(self, filter_by):
    #     profiletype = self["entitylinking"].get_benchmark_querytype()
    #     if profiletype == 'query':
    #         raise ValueError(f"query word filter cannot be used for querytype: query")
    #     if profiletype in ['basicprofile', 'chatprofile'] and filter_by == 'topic': #TODO???
    #         raise ValueError(
    #             f"domain_specific query filter cannot be used on {profiletype} (without a specified domain)")
    #
    #     if filter_by == 'amazon':
    #         G_tfs_raw, G_len_raw = DocStats.get_G_tfs_amazon_raw_from_file()
    #         corpus = "" #what? TODO
    #         doc = self["tokenizer"].tokenize(corpus)
    #         domain_counter = Counter(doc)
    #         G_probs = {k: (v + (G_tfs_raw[k] if k in G_tfs_raw else 0)) / (len(doc) + G_len_raw) for k, v in
    #                    domain_counter.items()}
    #         G_len = len(doc) + G_len_raw
    #         #we need G_probs
    #     elif filter_by == 'topic':
    #         pass
    #     elif filter_by == 'user':
    #         baseprofiletype = profiletype.split("_")[0] #chatprofile/basicprofile
    #         if baseprofiletype not in ["chatprofile", "basicprofile"]:
    #             ValueError(f"Baseprofile value is wrong {baseprofiletype}")
    #         profile_term_prob
    #
    #
    #
    #     # we need G_probs which contains our corpuses inside as well!
    #     # and the initial profile term probability which is different than quid_term_prob.
    #     # it should be more general. depending on what we are using it may be different
    #     reweighted_term_weights = {}
    #
    #     for term, p in profile_term_prob.items():
    #         reweighted_term_weights[term] = p / G_probs[term]
    #     return reweighted_term_weights

    def get_domain_specific_term_weights(self, corpus_name, tf_or_df, docids):
        if tf_or_df == 'tf':
            domain_term_probs = self.get_domain_term_probs_tf(docids)
            if corpus_name == "all_domains":
                G_probs = self.get_G_probs_all_corpus_tfs()
            elif corpus_name == 'amazon':
                G_probs = self.get_G_probs_amazon_tfs()
            else:
                raise ValueError(f"domain-term specific weighting not implemented for {corpus_name}")
        elif tf_or_df == 'df':
            domain_term_probs = self.get_domain_term_probs_df(docids)
            if corpus_name == "all_domains":
                G_probs = self.get_G_probs_all_corpus_dfs()
            elif corpus_name == 'amazon':
                G_probs = self.get_G_probs_amazon_dfs()
            else:
                raise ValueError(f"domain-term specific weighting not implemented for {corpus_name}")

        reweighted_term_weights = {}

        for term, p in domain_term_probs.items():
            reweighted_term_weights[term] = p / G_probs[term]

        # normalize : we could normalize them, but let's not...
#         sum_vals = sum(reweighted_term_weights[domain].values())
#         reweighted_term_weights[domain] = {k: v/sum_vals for k, v in reweighted_term_weights[domain].items()}

        return reweighted_term_weights

    def get_domain_term_probs_tf(self, docids):
        corpus = ""
        for docid in docids:
            corpus += self["index"].get_doc(docid)
            corpus += '\n'
        doc = self["tokenizer"].tokenize(corpus)
        doc_counter = Counter(doc)
        domain_term_probs = {k: (v / len(doc)) for k, v in doc_counter.items()}
        return domain_term_probs

    def get_domain_term_probs_df(self, docids):
        tokenized_docs = {}
        all_vocab = set()
        #I could directly use the index to get the df,... but I just used this for now. It doesn't take much time.
        for docid in docids:
            doc = self["tokenizer"].tokenize(self["index"].get_doc(docid))
            doc_counter = Counter(doc)
            all_vocab.update(doc_counter.keys())
            tokenized_docs[docid] = doc_counter.keys()
        dfs = {}
        for v in all_vocab:
            dfs[v] = 0
            for d in tokenized_docs:
                if v in tokenized_docs[d]:
                    dfs[v] += 1

        domain_probs = {k: (v / len(docids)) for k, v in dfs.items()}
        return domain_probs

    @staticmethod
    def getcontent(file):
        txt = []
        content = False
        with open(file) as f:
            for l in f:
                if l.strip().startswith("<TEXT>"):
                    content = True
                if l.strip().endswith("</TEXT>"):
                    content = False
                    l = l.replace("</TEXT>", '').strip()
                    if len(l) > 0:
                        txt.append(l)

                if content:
                    l = l.replace("<TEXT>", '').strip()
                    if len(l) > 0:
                        txt.append(l)

        txt = '\n'.join(txt)
        return txt

    @staticmethod
    def load_all_domains_corpus():
        domain_documents = {}

        for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
            doc_dir = f"/GW/PKB/work/data_personalization/TREC_format/documents/{domain}/"

            domain_documents[domain] = {}

            files = listdir(doc_dir)
            for fn in files:
                fid = fn[:-4]
                txt = DocStats.getcontent(join(doc_dir, fn))
                domain_documents[domain][fid] = txt
        return domain_documents

    @staticmethod
    def get_G_tfs_amazon_raw_from_file():
        amazonfile = PACKAGE_PATH / "data" / "corpus_stats" / "amazon_reviews_term_freq"
        if exists(amazonfile):
            G_tfs = pickle.load(open(amazonfile, "rb"))
            G_len = 0
            for v, tf in G_tfs.items():
                G_len += tf
            return G_tfs, G_len
        RuntimeError(f"{amazonfile} does not exist!")

    @staticmethod
    def get_G_dfs_amazon_raw_from_file():
        amazonfile = PACKAGE_PATH / "data" / "corpus_stats" / "amazon_reviews_doc_freq"
        if exists(amazonfile):
            data = pickle.load(open(amazonfile, "rb"))
            G_dfs = data["G_dfs"]
            G_num_docs = data["G_num_docs"]
            return G_dfs, G_num_docs
        RuntimeError(f"{amazonfile} does not exist!")

    def get_G_probs_all_corpus_tfs(self):
        all_docs = DocStats.load_all_domains_corpus()
        corpus = ""
        for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
            corpus += '\n'.join(all_docs[domain].values())
            corpus += '\n'

        doc = self["tokenizer"].tokenize(corpus)
        doc_counter = Counter(doc)
        G_probs = {k: (v / len(doc)) for k, v in doc_counter.items()}
        G_len = len(doc)
        return G_probs

    def get_G_probs_amazon_tfs(self):
        G_tfs_raw, G_len_raw = DocStats.get_G_tfs_amazon_raw_from_file()
        all_docs = DocStats.load_all_domains_corpus()
        corpus = ""
        for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
            corpus += '\n'.join(all_docs[domain].values())
            corpus += '\n'

        doc = self["tokenizer"].tokenize(corpus)
        domain_counter = Counter(doc)
        G_probs = {k: (v + (G_tfs_raw[k] if k in G_tfs_raw else 0)) / (len(doc) + G_len_raw) for k, v in domain_counter.items()}
        G_len = len(doc) + G_len_raw
        return G_probs

    def get_G_probs_amazon_dfs(self):
        G_dfs_raw, G_num_docs_raw = DocStats.get_G_dfs_amazon_raw_from_file()
        all_docs = DocStats.load_all_domains_corpus()

        d_num_docs = 0
        dfs = {}
        for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
            for d in all_docs[domain]:
                doc = self["tokenizer"].tokenize(all_docs[domain][d])
                for term in set(doc):
                    if term not in dfs:
                        dfs[term] = 0
                    dfs[term] += 1
                d_num_docs += 1

        G_num_docs = d_num_docs + G_num_docs_raw
        G_probs = {k: (v + (G_dfs_raw[k] if k in G_dfs_raw else 0)) / G_num_docs for k, v in dfs.items()}
        return G_probs
    
    def get_G_probs_all_corpus_dfs(self):
        all_docs = DocStats.load_all_domains_corpus()
        tokenized_docs = {}
        all_vocab = set()
        for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
            for d in all_docs[domain]:
                doc = self["tokenizer"].tokenize(all_docs[domain][d])
                doc_counter = Counter(doc)
                all_vocab.update(doc_counter.keys())
                tokenized_docs[f"{domain}_{d}"] = doc_counter.keys()
        dfs = {}
        for v in all_vocab:
            dfs[v] = 0
            for d in tokenized_docs:
                if v in tokenized_docs[d]:
                    dfs[v] += 1

        G_num_docs = len(tokenized_docs)
        G_probs = {k: (v / G_num_docs) for k, v in dfs.items()}
        return G_probs

    def get_user_general_corpus(self, qids):
        goutf = join(self.get_profile_term_prob_cache_path(), "allusers")
        if exists(goutf):
            with open(goutf, 'r') as f:
                GU = json.loads(f.read())
        else:
            user_profile_tfs = {}
            total_len = 0
            voc = set()
            for qid in qids:
                uid = qid.split("_")[1]
                if uid not in user_profile_tfs:
                    tfs = self.qid_termprob[qid]
                    mintf = min(tfs.values())
                    voc.update(tfs.keys())
                    profile_len = 1 / mintf
                    total_len += profile_len
                    user_profile_tfs[uid] = tfs  # just to have them with user id and uniquely
            GU = {}
            for v in voc:
                nu = 0
                for uid, tfs in user_profile_tfs.items():
                    if v in tfs:
                        nu += tfs[v]
                GU[v] = nu / total_len
            with open(goutf, 'w') as f:
                f.write(json.dumps(GU))

        return GU

    def get_domain_general_corpus(self, qids, baseprofiletype, userfullprofiles):
        GD = {}
        for qid in qids:
            uid = qid.split("_")[1]
            if uid not in GD:
                entoutf = join(self.get_selected_entities_cache_path(),
                               get_file_name(qid, self["entitylinking"].get_benchmark_name(), baseprofiletype))
                if exists(entoutf):
                    with open(entoutf, 'r') as f:
                        qentities = json.loads(f.read())
                else:
                    raise RuntimeError("This is not implemented! You should have already have the entities for the full profile in the cache to use this.")
                qdesc = []
                for e in qentities["NE"]:
                    qdesc.append(self["entitylinking"].get_entity_description(e))
                for e in qentities["C"]:
                    qdesc.append(self["entitylinking"].get_entity_description(e))

                qtext = userfullprofiles[uid]
                qtext += "\n" + "\n".join(qdesc)
                query = self["tokenizer"].tokenize(qtext)

                q_count = Counter(query)
                GD[uid] = {k: (v / len(query)) for k, v in q_count.items()}

        return GD

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

    def id2vec(self, qid, posid, negid=None, query=None):#todo (ask) where is it used?
        # if query is not None:
        #     if qid is None:
        #         query = self["tokenizer"].tokenize(query)
        #     else:
        #         raise RuntimeError("received both a qid and query, but only one can be passed")
        # else:
        #     query = self.qid_termprob[qid]

        return {"qid": qid, "posdocid": posid, "negdocid": negid}


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
        filter_query = None

        if entity_strategy not in [None, 'all', 'domain', 'specific_domainrel']:  # TODO add strategies
            raise ValueError(f"invalid entity usage strategy (or not implemented): {entity_strategy}")

        if filter_query is not None and not re.match(r"^(domain|user)_specific_k(\d+|-1)$", filter_query):
            raise ValueError(f"invalid filter query: {filter_query}")

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

    def id2vec(self, qid, posid, negid=None, query=None):#todo change this later or delete it...
        # if query is not None:
        #     if qid is None:
        #         query = self["tokenizer"].tokenize(query)
        #     else:
        #         raise RuntimeError("received both a qid and query, but only one can be passed")
        # else:
        #     query = self.qid2toks[qid]

        return {"qid": qid, "posdocid": posid, "negdocid": negid}
