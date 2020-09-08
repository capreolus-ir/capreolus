import os
import pickle
from collections import defaultdict

import pytest
from tqdm import tqdm

from capreolus import Benchmark, module_registry
from capreolus.utils.common import download_file
from capreolus.utils.trec import load_qrels

from capreolus.benchmark.codesearchnet import CodeSearchNetChallenge as CodeSearchNetCodeSearchNetChallengeBenchmark
from capreolus.benchmark.codesearchnet import CodeSearchNetCorpus as CodeSearchNetCodeSearchNetCorpusBenchmark
from capreolus.collection.codesearchnet import CodeSearchNet as CodeSearchNetCollection
from capreolus.collection.covid import COVID as CovidCollection
from capreolus.benchmark.covid import COVID as CovidBenchmark

from capreolus.tests.common_fixtures import tmpdir_as_cache
from capreolus.utils.common import remove_newline
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

benchmarks = set(module_registry.get_module_names("benchmark"))


@pytest.mark.parametrize("benchmark_name", benchmarks)
@pytest.mark.download
def test_benchmark_creatable(tmpdir_as_cache, benchmark_name):
    benchmark = Benchmark.create(benchmark_name)
    if hasattr(benchmark, "download_if_missing"):
        benchmark.download_if_missing()


@pytest.mark.download
def test_csn_corpus_benchmark_downloadifmissing():
    for lang in ["ruby"]:
        logger.info(f"testing {lang}")
        cfg = {"name": "codesearchnet_corpus", "lang": lang}
        benchmark = CodeSearchNetCodeSearchNetCorpusBenchmark(cfg)
        benchmark.download_if_missing()

        assert os.path.exists(benchmark.docid_map_file)
        assert os.path.exists(benchmark.qid_map_file)

        assert os.path.exists(benchmark.topic_dir / f"{lang}.txt")
        assert os.path.exists(benchmark.qrel_dir / f"{lang}.txt")
        assert os.path.exists(benchmark.fold_dir / f"{lang}.json")


def _load_trec_doc(fn):
    id2doc = {}
    with open(fn, "r", encoding="utf-8") as f:
        docno, doc = None, ""
        read_doc = False
        for line in f:
            if line.startswith("<DOCNO>"):
                docno = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                continue

            if line.strip() == "</DOC>":
                id2doc[docno] = doc
                docno, doc = None, ""
                continue

            if line.strip() == "<TEXT>":
                read_doc = True
                continue

            if line.strip() == "</TEXT>":
                read_doc = False
                continue

            if read_doc:
                doc += line.strip()
    return id2doc


@pytest.mark.download
def test_csn_coll_benchmark_consistency():
    for lang in ["ruby"]:
        cfg = {"name": "codesearchnet_corpus", "lang": lang}
        benchmark = CodeSearchNetCodeSearchNetCorpusBenchmark(cfg)
        collection = CodeSearchNetCollection(cfg)

        pkl_path = collection.get_cache_path() / "tmp" / f"{lang}_dedupe_definitions_v2.pkl"  # TODO: how to remove this "hack"
        coll_path = os.path.join(collection.download_if_missing(), f"csn-{lang}-collection.txt")

        raw_data = pickle.load(open(pkl_path, "rb"))
        doc_coll = _load_trec_doc(coll_path)
        assert len(raw_data) == len(doc_coll)

        sorted_docnos = sorted(doc_coll.keys(), key=lambda x: int(x.split("-")[-1]))
        logger.debug(f"sorted keys: {sorted_docnos[:10]}")

        total = len(raw_data)
        for raw_doc, docno in tqdm(zip(raw_data, sorted_docnos), total=total):
            doc = doc_coll[docno]
            code_tokens = remove_newline(" ".join(raw_doc["function_tokens"]))
            benc_docno = benchmark.get_docid(raw_doc["url"], code_tokens)

            assert doc == code_tokens
            assert docno == benc_docno


@pytest.mark.download
def test_csn_challenge_download_if_missing():
    config = {"name": "codesearchnet_challenge", "lang": "ruby"}
    benmchmark = CodeSearchNetCodeSearchNetChallengeBenchmark(config)
    benmchmark.download_if_missing()

    assert benmchmark.qid_map_file.exists() and benmchmark.topic_file.exists()


@pytest.mark.download
def test_covid_round3_qrel_conversion():
    collection_config = {"name": "covid", "round": 3, "coll_type": "abstract"}
    benchmark_config = {"name": "covid", "udelqexpand": False, "useprevqrels": False}
    collection = CovidCollection(collection_config)
    benchmark = CovidBenchmark(benchmark_config, provide={"collectoin": collection})

    benchmark.download_if_missing()

    docid_map_tmp = "/tmp/docid.map"
    newdocid_qrels_fn = "/tmp/new.docid.qrels"
    qrel_url = "https://ir.nist.gov/covidSubmit/data/qrels-covid_d3_j0.5-3.txt"
    docid_map_url = "https://ir.nist.gov/covidSubmit/data/changedIds-May19.csv"

    download_file(docid_map_url, docid_map_tmp)
    download_file(qrel_url, newdocid_qrels_fn)
    with open(docid_map_tmp) as f:
        old2new = {line.split(",")[0]: line.split(",")[1] for line in f}
    newdocid_qrels = load_qrels(newdocid_qrels_fn)
    olddocid_qrels = benchmark.qrels

    # since there are dropped out terms in benchmark.qrels (the ones that appeared in previous judgements)
    # converted olddocid_qrels will have less entries than newdocid_qrels.
    # Cannot use assert convert_qrels == newdocid_qrels here
    for qid in olddocid_qrels:
        for docid in olddocid_qrels[qid]:
            newdocid = old2new.get(docid, docid)
            assert olddocid_qrels[qid][docid] == newdocid_qrels[qid][newdocid]
