import os
import pickle
from tqdm import tqdm

from capreolus.utils.loginit import get_logger
from capreolus.utils.common import remove_newline
from capreolus.benchmark import CodeSearchNet as CodeSearchNetBenchmark
from capreolus.collection import CodeSearchNet as CodeSearchNetCollection

logger = get_logger(__name__)


def test_csn_corpus_benchmark_downloadifmissing():
    for lang in ["python", "java", "javascript", "go", "ruby", "php"]:
        logger.info(f"testing {lang}")
        cfg = {"_name": "codesearchnet", "lang": lang}
        benchmark = CodeSearchNetBenchmark(cfg)
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


def test_csn_coll_benchmark_consistency():
    for lang in ["python", "java", "javascript", "go", "ruby", "php"]:
        cfg = {"_name": "codesearchnet", "lang": lang}
        benchmark = CodeSearchNetBenchmark(cfg)
        collection = CodeSearchNetCollection(cfg)

        pkl_path = collection.get_cache_path() / "tmp" / f"{lang}_dedupe_definitions_v2.pkl"  # TODO: how to remove this "hack"
        coll_path = collection.download_if_missing() / f"csn-{lang}-collection.txt"

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
