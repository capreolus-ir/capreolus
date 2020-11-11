import pickle
from os.path import join, exists
from os import listdir
import re

from capreolus.utils.common import load_trec_topics
from capreolus.registry import PACKAGE_PATH

def load_all_domains_corpus():
    domain_documents = {}

    for domain in ['movie', 'travel_wikivoyage', 'food', 'book']:
        doc_dir = f"/GW/PKB/work/data_personalization/TREC_format_quselection_C_final_profiles/documents/{domain}/"

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
            return re.sub(r'(\d+)_(.+)', r'\g<1>', fid)
        else:
            return re.sub(r'(\d+)_(.+)', r'\g<2>', fid)
    else:
        return fid