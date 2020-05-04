#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:2/5/2020

import os
import re
import gzip
import json

from tqdm import tqdm
from argparse import ArgumentParser

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]


def remove_newline(txt):
    return txt.replace("\r", "").replace("\n", "").strip()


def get_camel_parser():
    camel_patterns = [re.compile('(.)([A-Z][a-z]+)'), re.compile('([a-z0-9])([A-Z])')]

    def camel_parser(name):
        for pattern in camel_patterns:
            name = pattern.sub(r'\1 \2', name)
        return name.lower()

    return camel_parser


class Docobj2docid:
    def __init__(self, docmap_fn):
        self.docmap = json.load(open(docmap_fn))

    def __getitem__(self, item):
        url, code_tokens = item["url"], " ".join(item["code_tokens"])
        docids = self.docmap[url]
        return docids[0] if len(docids) == 1 else docids[code_tokens]


def parse_lang(lang):
    """ lang: str """
    return LANGS if lang == "all" else [lang]


def prep_csn_runfile(csn_rawdata_dir, map_dir, langs, csn_outp_dir, withcamel=True):
    config_name = "with_camelstem" if withcamel else "without_camelstem"

    """ chunk the downloaded csn gzip file into size of 1k """
    for lang in langs:
        csn_lang_dir = os.path.join(csn_rawdata_dir, lang, "final", "jsonl")
        outp_lang_dir = os.path.join(csn_outp_dir, lang)
        os.makedirs(outp_lang_dir, exist_ok=True)

        qidmap = json.load(open(os.path.join(map_dir, "qidmap", config_name, f"{lang}.json")))
        q_keys = list(qidmap.keys())
        for k in q_keys:
            qidmap[" ".join(k.split())] = qidmap.pop(k)
        docidmap = Docobj2docid(docmap_fn=os.path.join(map_dir, "docidmap", config_name, f"{lang}.json"))

        camel_parser = get_camel_parser()
        # for set_name in ["train", "valid", "test"]:
        for set_name in ["test"]:
            csn_lang_path = os.path.join(csn_lang_dir, set_name)
            outp_lang_path = os.path.join(outp_lang_dir, f"{set_name}.runfile.txt")
            outp_f = open(outp_lang_path, "w")

            objs = []
            # for file in tqdm(os.listdir(csn_lang_path), desc=f"processing {lang} - {set_name}"):
            for file in os.listdir(csn_lang_path):
                print(f"processing {lang} - {set_name}")
                if not file.endswith("jsonl.gz"):
                    continue

                with gzip.open(os.path.join(csn_lang_path, file)) as f:
                    lines = f.readlines()
                    for line in tqdm(lines, desc=f"reading file {file}"):
                        objs.append(json.loads(line))

                        if len(objs) == 1000:
                            for obj1 in objs:
                                gt_docid, all_docs = docidmap[obj1], []
                                docstring = remove_newline(" ".join(obj1["docstring_tokens"]))
                                docstring = camel_parser(docstring).replace("_", " ").strip() if withcamel else docstring # tmp
                                docstring = " ".join(docstring.split()[:1020])  # for TooManyClause
                                qid = qidmap[docstring]

                                for fake_rank, obj2 in enumerate(objs):
                                    docid = docidmap[obj2]
                                    all_docs.append(docid)

                                    if docid == gt_docid:
                                        fake_rank = 2000
                                    outp_f.write(f"{qid} Q0 {docid} {fake_rank} {(1000-fake_rank)/1000} csn\n")  # 0 Q0 go-FUNCTION-351245 1 9.159600 Anserini
                                assert gt_docid in all_docs
                            objs = []  # reset
                        # if the number of objects cannot be devided by 1000, ignore the rest part


def load_runfile(p):
    runs = {}
    with open(p) as f:
        for l in f:
            qid, _, docid, rank, _, _ = l.strip().split()
            if qid in runs:
                # runs[qid].append(docid)
                runs[qid][docid] = rank
            else:
                # runs[qid] = [docid]
                runs[qid] = {docid: rank}
    return runs


def filter_results(bm25_runfile_pattern, csn_runfile_dir, langs, outp_dir):
    """ for now only do the filtering on test dataset """
    for lang in langs:
        csn_lang_dir = os.path.join(csn_runfile_dir, lang)
        outp_lang_dir = os.path.join(outp_dir, lang)
        os.makedirs(outp_lang_dir, exist_ok=True)

        for set_name in ["test"]:
            bm25_lang_path = bm25_runfile_pattern % lang
            outp_lang_dir = os.path.join(outp_lang_dir, f"{set_name}.filtered.runfile")

            csn_lang_path = os.path.join(csn_lang_dir, f"{set_name}.runfile.txt")  # used as filter
            runs = load_runfile(csn_lang_path)

            # tmp
            gts = {q: [d for d in runs[q] if runs[q][d] == "2000"] for q in runs}
            found_docs = {}
            # end of tmp

            with open(bm25_lang_path) as f, open(outp_lang_dir, "w") as fout:
                for l in f:
                    qid, _, docid, _, _, _ = l.strip().split()

                    if qid not in runs:
                        continue

                    assert gts.get(qid, None)
                    if docid not in runs[qid]:
                        # print(docid, len(runs[qid]), "qid; ", qid)
                        continue

                    fout.write(l)
                    # tmp
                    if qid not in found_docs:
                        found_docs[qid] = [docid]
                    else:
                        found_docs[qid].append(docid)

        print(f"finished {lang}")
        # tmp
        print(f"number of recorded qid: {len(found_docs)}")
        doclens = [len(d) for d in found_docs.values()]
        print(f"doc statistics: ", max(doclens), min(doclens), sum(doclens)/len(doclens))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--overwrite_csn_runfile", "-o", type=bool, default=False)
    parser.add_argument("--lang", "-l", type=str, default="all")
    parser.add_argument("--withcamel", type=bool, default=False)

    parser.add_argument("--raw_csn_data", type=str, default="/tmp")
    parser.add_argument("--csn_runfile_dir", type=str, default="./csn_runfile_4")
    parser.add_argument(
        "--bm25_runfile_pattern", "-csn", type=str,
        # default="/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camel_parser_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.75_hits-100_k1-1.2/codesearchnet_corpus_camel_fix/searcher")
        # default="/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_lang-%s_camelstemmer-True/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_hits-100_k1-0.9/codesearchnet_corpus_camelstemmer-True/searcher")
        default="/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camelstemmer-True_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.75_hits-100_k1-1.2/codesearchnet_corpus/searcher")
    parser.add_argument(
        "--map_dir", type=str,
        default="/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_corpus")
        # default="/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_corpus_camel")

    args = parser.parse_args()
    langs = parse_lang(args.lang)

    csn_runfile_dir = os.path.join(args.csn_runfile_dir, "neighbour1k")
    csn_filtered_runfile_dir = os.path.join(args.csn_runfile_dir, "filtered_bm25")

    if args.overwrite_csn_runfile or \
        (args.lang == "all" and not os.path.exists(csn_runfile_dir)) or \
        (args.lang != "all" and not os.path.exists(os.path.join(csn_runfile_dir, args.lang))):
        os.makedirs(csn_runfile_dir, exist_ok=True)
        prep_csn_runfile(
            csn_rawdata_dir=args.raw_csn_data, map_dir=args.map_dir,
            langs=langs, csn_outp_dir=csn_runfile_dir, withcamel=args.withcamel)

    os.makedirs(csn_filtered_runfile_dir, exist_ok=True)
    filter_results(
        bm25_runfile_pattern=args.bm25_runfile_pattern, csn_runfile_dir=csn_runfile_dir,
        langs=langs, outp_dir=csn_filtered_runfile_dir)
