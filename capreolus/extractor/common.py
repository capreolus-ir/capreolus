import json
import pickle
from os.path import join, exists
from collections import Counter
from os import listdir
import re

from capreolus.utils.common import load_trec_topics
from capreolus.registry import PACKAGE_PATH

def load_all_domains_corpus(all_domains):
    domain_documents = {}

    for domain in all_domains:
        doc_dir = open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip() + "documents" # TODO this can be changed to get the collection's doc path, after rebasing and stuff

        domain_documents[domain] = {}

        files = listdir(doc_dir)
        for fn in files:
            fid = fn[:-4]
            txt = getcontent(join(doc_dir, fn))
            domain_documents[domain][fid] = txt
    return domain_documents

def get_G_tfs_amazon_raw_from_file():
    amazonfile = PACKAGE_PATH / "data" / "corpus_stats" / "amazon_reviews_term_freq"
    if exists(amazonfile):
        G_tfs = pickle.load(open(amazonfile, "rb"))
        G_len = 0
        for v, tf in G_tfs.items():
            G_len += tf
        return G_tfs, G_len
    RuntimeError(f"{amazonfile} does not exist!")

def get_G_dfs_amazon_raw_from_file():
    amazonfile = PACKAGE_PATH / "data" / "corpus_stats" / "amazon_reviews_doc_freq"
    if exists(amazonfile):
        data = pickle.load(open(amazonfile, "rb"))
        G_dfs = data["G_dfs"]
        G_num_docs = data["G_num_docs"]
        return G_dfs, G_num_docs
    RuntimeError(f"{amazonfile} does not exist!")

def get_all_user_profiles(queryfn):
    topics = load_trec_topics(queryfn)['title']
    profiles = {}
    for quid in topics:
        uid = quid.split("_")[-1]
        if uid not in profiles:
            profiles[uid] = topics[quid]

    return profiles

def get_amazon_plus_user_profile_term_probs_tf(G_tfs_raw, G_len_raw, profile_tfs, profile_len):
    total_len = profile_len + G_len_raw

    G_probs = {}
    for term, tf in profile_tfs.items():
        nu = tf
        if term in G_tfs_raw:
            nu += G_tfs_raw[term]
        G_probs[term] = nu / total_len

    return G_probs

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

def get_file_name(fid, benchmark_name, benchmark_querytype):
    ### This is written wrt our benchmarks and the ids we have for the queries.
    ### Maybe need to be extended on new benchmarks.
    ## The idea is that, we don't want to have redundency in the extraction and caching

    if benchmark_name in ['pes20', 'kitt']:
        if benchmark_querytype == "query":
            return re.sub(r'(.+)_(\d+)_(.+)', r'\g<1>\g<2>', fid)
        else:
            return re.sub(r'(.+)_(\d+)_(.+)', r'\g<3>', fid)
    else:
        return fid


### made static:
def get_all_users_profiles_term_frequency(profiletype, qids, tokenizer, docstat=False,
                                          entitycachepath=None, entitylinking=None):
    benchmarkdir = open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip() + "topics"  # TODO change after rebasing
    userfullprofiles = get_all_user_profiles(join(benchmarkdir, f"alldomains_topics.{profiletype}.txt"))

    user_profile_tfs = {}
    user_profile_len = {}
    total_len = 0
    voc = set()
    for qid in qids:
        uid = qid.split("_")[-1]
        if uid not in user_profile_tfs:
            #TODO entities are not incorporated to textembed extractor,
            # we just have them for the Docstat. Rm this if if it was added
            qdesc = []
            if docstat:
                entoutf = join(entitycachepath,
                               get_file_name(qid, entitylinking.get_benchmark_name(), profiletype))
                if exists(entoutf):
                    with open(entoutf, 'r') as f:
                        qentities = json.loads(f.read())
                else:
                    raise RuntimeError(
                        "This is not implemented! You should have already have the entities for the full profile in the cache to use this. To this end, you need to run it once for fold1 for example.")

                for e in qentities["NE"]:
                    qdesc.append(entitylinking.get_entity_description(e))
                for e in qentities["C"]:
                    qdesc.append(entitylinking.get_entity_description(e))

            qtext = userfullprofiles[uid]
            qtext += "\n" + "\n".join(qdesc)
            query = tokenizer.tokenize(qtext)
            q_count = Counter(query)
            user_profile_tfs[uid] = q_count
            user_profile_len[uid] = len(query)
            total_len += len(query)
            voc.update(q_count.keys())

    return voc, user_profile_tfs, total_len, user_profile_len


def get_all_users_profile_term_probs_tf(profiletype, qids, tokenizer, docstat=False,
                                          entitycachepath=None, entitylinking=None):
    voc, user_profile_tfs, total_len, _ = get_all_users_profiles_term_frequency(profiletype, qids, tokenizer, docstat, entitycachepath, entitylinking)
    allusers_term_probs = {}
    for term in voc:
        nu = 0
        for uid, tfs in user_profile_tfs.items():
            if term in tfs:
                nu += tfs[term]
        allusers_term_probs[term] = nu / total_len
    return allusers_term_probs


def get_amazon_plus_all_users_profile_term_probs_tf(profiletype, qids, tokenizer, docstat=False,
                                                        entitycachepath=None, entitylinking=None):
    voc, user_profile_tfs, profs_len, _ = get_all_users_profiles_term_frequency(profiletype, qids, tokenizer, docstat, entitycachepath, entitylinking)
    G_tfs_raw, G_len_raw = get_G_tfs_amazon_raw_from_file()
    total_len = profs_len + G_len_raw

    G_probs = {}
    for term in voc:
        nu = 0
        for uid, tfs in user_profile_tfs.items():
            if term in tfs:
                nu += tfs[term]
        if term in G_tfs_raw:
            nu += G_tfs_raw[term]

        G_probs[term] = nu / total_len

    return G_probs


def get_domain_term_probabilities_tf(docids, index, tokenizer):
    corpus = ""
    for docid in docids:
        corpus += index.get_doc(docid)
        corpus += '\n'
    doc = tokenizer.tokenize(corpus)
    doc_counter = Counter(doc)
    domain_term_probabilities = {k: (v / len(doc)) for k, v in doc_counter.items()}
    return domain_term_probabilities


def get_G_probabilities_all_corpus_tfs(all_domains, tokenizer):
    all_docs = load_all_domains_corpus(all_domains)
    corpus = ""
    for domain in all_domains:
        corpus += '\n'.join(all_docs[domain].values())
        corpus += '\n'

    doc = tokenizer.tokenize(corpus)
    doc_counter = Counter(doc)
    G_probabilities = {k: (v / len(doc)) for k, v in doc_counter.items()}
    G_len = len(doc)
    return G_probabilities


def get_G_probabilities_amazon_tfs(all_domains, tokenizer):
    G_tfs_raw, G_len_raw = get_G_tfs_amazon_raw_from_file()
    all_docs = load_all_domains_corpus(all_domains)
    corpus = ""
    for domain in all_domains:
        corpus += '\n'.join(all_docs[domain].values())
        corpus += '\n'

    doc = tokenizer.tokenize(corpus)
    domain_counter = Counter(doc)
    G_probs = {k: (v + (G_tfs_raw[k] if k in G_tfs_raw else 0)) / (len(doc) + G_len_raw) for k, v in domain_counter.items()}
    G_len = len(doc) + G_len_raw
    return G_probs


def get_domain_term_probabilities_df(docids, index, tokenizer):
    tokenized_docs = {}
    all_vocab = set()
    #I could directly use the index to get the df,... but I just used this for now. It doesn't take much time.
    for docid in docids:
        doc = tokenizer.tokenize(index.get_doc(docid))
        doc_counter = Counter(doc)
        all_vocab.update(doc_counter.keys())
        tokenized_docs[docid] = doc_counter.keys()
    dfs = {}
    for v in all_vocab:
        dfs[v] = 0
        for d in tokenized_docs:
            if v in tokenized_docs[d]:
                dfs[v] += 1

    domain_probabilities = {k: (v / len(docids)) for k, v in dfs.items()}
    return domain_probabilities


def get_G_probabilities_all_corpus_dfs(all_domains, tokenizer):
    all_docs = load_all_domains_corpus(all_domains)
    tokenized_docs = {}
    all_vocab = set()
    for domain in all_domains:
        for d in all_docs[domain]:
            doc = tokenizer.tokenize(all_docs[domain][d])
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


def get_G_probabilities_amazon_dfs(all_domains, tokenizer):
    G_dfs_raw, G_num_docs_raw = get_G_dfs_amazon_raw_from_file()
    all_docs = load_all_domains_corpus(all_domains)

    d_num_docs = 0
    dfs = {}
    for domain in all_domains:
        for d in all_docs[domain]:
            doc = tokenizer.tokenize(all_docs[domain][d])
            for term in set(doc):
                if term not in dfs:
                    dfs[term] = 0
                dfs[term] += 1
            d_num_docs += 1

    G_num_docs = d_num_docs + G_num_docs_raw
    G_probs = {k: (v + (G_dfs_raw[k] if k in G_dfs_raw else 0)) / G_num_docs for k, v in dfs.items()}
    return G_probs


def get_domain_specific_term_weights(corpus_name, tf_or_df, docids,
                                     all_domains, index, tokenizer):
    if tf_or_df == 'tf':
        domain_term_probs = get_domain_term_probabilities_tf(docids, index, tokenizer)
        if corpus_name == "all_domains":
            G_probs = get_G_probabilities_all_corpus_tfs(all_domains, tokenizer)
        elif corpus_name == 'amazon':
            G_probs = get_G_probabilities_amazon_tfs(all_domains, tokenizer)
        else:
            raise ValueError(f"domain-term specific weighting not implemented for {corpus_name}")
    elif tf_or_df == 'df':
        domain_term_probs = get_domain_term_probabilities_df(docids, index, tokenizer)
        if corpus_name == "all_domains":
            G_probs = get_G_probabilities_all_corpus_dfs(all_domains, tokenizer)
        elif corpus_name == 'amazon':
            G_probs = get_G_probabilities_amazon_dfs(all_domains, tokenizer)
        else:
            raise ValueError(f"domain-term specific weighting not implemented for {corpus_name}")

    term_weights = {}

    for term, p in domain_term_probs.items():
        term_weights[term] = p / G_probs[term]

    # normalize : we could normalize them, but let's not...
    #         sum_vals = sum(reweighted_term_weights[domain].values())
    #         reweighted_term_weights[domain] = {k: v/sum_vals for k, v in reweighted_term_weights[domain].items()}

    return term_weights


def get_profile_term_weight_topic(qids, profiletype, query_term_weighting_strategy, query_term_weighting_strategy_corpus,
                                  tokenizer, docstat=False, entitycachepath=None, entitylinking=None):
    if profiletype in ['basicprofile', 'chatprofile', 'query']:
        raise ValueError(f"{query_term_weighting_strategy} query word weighting/cut cannot be used for querytype: {profiletype}")

    # profiletype is topical. s_probs contains term probs given a topic
    s_probabilities = get_all_users_profile_term_probs_tf(profiletype, qids, tokenizer, docstat, entitycachepath, entitylinking)
    if query_term_weighting_strategy_corpus == 'alltopics':
        baseprofiletype = "chatprofile" if profiletype.startswith("chatprofile") else "basicprofile"
        G_probs = get_all_users_profile_term_probs_tf(baseprofiletype, qids, tokenizer, docstat, entitycachepath, entitylinking)
    elif query_term_weighting_strategy_corpus == 'amazon':
        baseprofiletype = "chatprofile" if profiletype.startswith("chatprofile") else "basicprofile"
        G_probs = get_amazon_plus_all_users_profile_term_probs_tf(baseprofiletype, qids, tokenizer, docstat, entitycachepath, entitylinking)

    term_weights = {}
    for term, p in s_probabilities.items():
        term_weights[term] = p / G_probs[term]
    return term_weights

# self["tokenizer"], True, self.get_selected_entities_cache_path(), self["entitylinking"]
# profiletype = self["entitylinking"].get_benchmark_querytype()
def get_profile_term_weight_user(qids, profiletype, query_term_weighting_strategy, query_term_weighting_strategy_corpus,
                                 tokenizer, docstat=False, entitycachepath=None, entitylinking=None):
    if profiletype == 'query':
        raise ValueError(f"{query_term_weighting_strategy} query word weighting/cut cannot be used for querytype: {profiletype}")

    baseprofiletype = "chatprofile" if profiletype.startswith("chatprofile") else "basicprofile"
    voc, user_profile_tfs, total_len, user_profile_len = get_all_users_profiles_term_frequency(baseprofiletype, qids, tokenizer,
                                                                                               docstat, entitycachepath, entitylinking)
    s_user_probabilities = {}
    for uid in user_profile_tfs:
        s_user_probabilities[uid] = {}
        for term, tf in user_profile_tfs[uid].items():
            s_user_probabilities[uid][term] = tf / user_profile_len[uid]

    if query_term_weighting_strategy_corpus == 'allusers':
        G_probs = {}
        for term in voc:
            nu = 0
            for uid, tfs in user_profile_tfs.items():
                if term in tfs:
                    nu += tfs[term]
            G_probs[term] = nu / total_len
        user_term_weights = {}
        for uid in s_user_probabilities:
            user_term_weights[uid] = {}
            for term, p in s_user_probabilities[uid].items():
                user_term_weights[uid][term] = p / G_probs[term]
        return user_term_weights
    elif query_term_weighting_strategy_corpus == 'amazon':
        G_tfs_raw, G_len_raw = get_G_tfs_amazon_raw_from_file()
        user_term_weights = {}
        for uid in s_user_probabilities.keys():
            user_term_weights[uid] = {}
            G_probs = get_amazon_plus_user_profile_term_probs_tf(G_tfs_raw, G_len_raw, user_profile_tfs[uid], user_profile_len[uid])
            for term, p in s_user_probabilities[uid].items():
                user_term_weights[uid][term] = p / G_probs[term]
        return user_term_weights
