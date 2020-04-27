import json
import os
import gzip
import pickle

from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels, load_trec_topics, topic_to_trectxt
from capreolus.utils.common import download_file, hash_file, remove_newline

logger = get_logger(__name__)


class Benchmark(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "benchmark"
    qrel_file = None
    topic_file = None
    fold_file = None
    query_type = None

    @property
    def qrels(self):
        if not hasattr(self, "_qrels"):
            self._qrels = load_qrels(self.qrel_file)
        return self._qrels

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.topic_file)
        return self._topics

    @property
    def folds(self):
        if not hasattr(self, "_folds"):
            self._folds = json.load(open(self.fold_file, "rt"))
        return self._folds


class DummyBenchmark(Benchmark):
    name = "dummy"
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"


class WSDM20Demo(Benchmark):
    name = "wsdm20demo"
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


class Robust04Yang19(Benchmark):
    name = "robust04.yang19"
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


class ANTIQUE(Benchmark):
    name = "antique"
    qrel_file = PACKAGE_PATH / "data" / "qrels.antique.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.antique.txt"
    fold_file = PACKAGE_PATH / "data" / "antique.json"
    query_type = "title"


class MSMarcoPassage(Benchmark):
    name = "msmarcopassage"
    qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcopassage.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.msmarcopassage.txt"
    fold_file = PACKAGE_PATH / "data" / "msmarcopassage.folds.json"
    query_type = "title"


class CodeSearchNetCorpus(Benchmark):
    name = "codesearchnet_corpus"
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    query_type = "title"

    file_fn = PACKAGE_PATH / "data" / "csn_corpus"

    qrel_dir = file_fn / "qrels"
    topic_dir = file_fn / "topics"
    fold_dir = file_fn / "folds"

    qidmap_dir = file_fn / "qidmap"
    docidmap_dir = file_fn / "docidmap"

    @staticmethod
    def config():
        lang = "ruby"  # which language dataset under CodeSearchNet

    def __init__(self, cfg):
        super().__init__(cfg)
        lang = cfg["lang"]

        self.qid_map_file = self.qidmap_dir / f"{lang}.json"
        self.docid_map_file = self.docidmap_dir / f"{lang}.json"

        self.qrel_file = self.qrel_dir / f"{lang}.txt"
        self.topic_file = self.topic_dir / f"{lang}.txt"
        self.fold_file = self.fold_dir / f"{lang}.json"

        for file in [var for var in vars(self) if var.endswith("file")]:
            eval(f"self.{file}").parent.mkdir(exist_ok=True, parents=True)  # TODO: is eval a good coding habit?

        self.download_if_missing()

    @property
    def qid_map(self):
        if not hasattr(self, "_qid_map"):
            if not self.qid_map_file.exists():
                self.download_if_missing()

            self._qid_map = json.load(open(self.qid_map_file, "r"))
        return self._qid_map

    @property
    def docid_map(self):
        if not hasattr(self, "_docid_map"):
            if not self.docid_map_file.exists():
                self.download_if_missing()

            self._docid_map = json.load(open(self.docid_map_file, "r"))
        return self._docid_map

    def download_if_missing(self):
        files = [self.qid_map_file, self.docid_map_file, self.qrel_file, self.topic_file, self.fold_file]
        if all([f.exists() for f in files]):
            return

        lang = self.cfg["lang"]

        tmp_dir = Path("/tmp")
        zip_fn = tmp_dir / f"{lang}.zip"
        if not zip_fn.exists():
            download_file(f"{self.url}/{lang}.zip", zip_fn)

        with ZipFile(zip_fn, "r") as zipobj:
            zipobj.extractall(tmp_dir)

        # prepare docid-url mapping from dedup.pkl
        pkl_fn = tmp_dir / f"{lang}_dedupe_definitions_v2.pkl"
        doc_objs = pickle.load(open(pkl_fn, "rb"))
        self._docid_map = self._prep_docid_map(doc_objs)
        assert self._get_n_docid() == len(doc_objs)

        # prepare folds, qrels, topics, docstring2qid  # TODO: shall we add negative samples?
        qrels, self._qid_map = defaultdict(dict), {}
        qids = {s: [] for s in ["train", "valid", "test"]}

        topic_file = open(self.topic_file, "w", encoding="utf-8")
        qrel_file = open(self.qrel_file, "w", encoding="utf-8")

        def gen_doc_from_gzdir(dir):
            """ generate parsed dict-format doc from all jsonl.gz files under given directory """
            for fn in sorted(dir.glob("*.jsonl.gz")):
                f = gzip.open(fn, "rb")
                for doc in f:
                    yield json.loads(doc)

        for set_name in qids:
            set_path = tmp_dir / lang / "final" / "jsonl" / set_name
            for doc in gen_doc_from_gzdir(set_path):
                code = remove_newline(" ".join(doc["code_tokens"]))
                docstring = remove_newline(" ".join(doc["docstring_tokens"]))
                n_words_in_docstring = len(docstring.split())
                if n_words_in_docstring >= 1024:
                    logger.warning(f"chunk query to first 1000 words otherwise TooManyClause would be triggered "
                                   f"at lucene at search stage, ")
                    docstring = " ".join(docstring.split()[:1020])  # for TooManyClause

                docid = self.get_docid(doc["url"], code)
                qid = self._qid_map.get(docstring, str(len(self._qid_map)))
                qrel_file.write(f"{qid} Q0 {docid} 1\n")

                if docstring not in self._qid_map:
                    self._qid_map[docstring] = qid
                    qids[set_name].append(qid)
                    topic_file.write(topic_to_trectxt(qid, docstring))

        topic_file.close()
        qrel_file.close()

        # write to qid_map.json, docid_map, fold.json
        json.dump(self._qid_map, open(self.qid_map_file, "w"))
        json.dump(self._docid_map, open(self.docid_map_file, "w"))
        json.dump({"s1": {
            "train": qids["train"],
            "predict": {"dev": qids["valid"], "test": qids["test"]}
        }}, open(self.fold_file, "w"))

    def _prep_docid_map(self, doc_objs):
        """
        construct a nested dict to map each doc into a unique docid
        which follows the structure: {url: {" ".join(code_tokens): docid, ...}}

        For all the lanugage datasets the url uniquely maps to a code_tokens yet it's not the case for but js and php
        which requires a second-level mapping from raw_doc to docid

        :param doc_objs: a list of dict having keys ["nwo", "url", "sha", "identifier", "arguments"
            "function", "function_tokens", "docstring", "doctring_tokens",],
        :return:
        """
        # TODO: any way to avoid the twice traversal of all url and make the return dict structure consistent
        lang = self.cfg["lang"]
        url2docid = defaultdict(dict)
        for i, doc in tqdm(enumerate(doc_objs), desc=f"Preparing the {lang} docid_map"):
            url, code_tokens = doc["url"], remove_newline(" ".join(doc["function_tokens"]))
            url2docid[url][code_tokens] = f"{lang}-FUNCTION-{i}"

        # remove the code_tokens for the unique url-docid mapping
        for url, docids in tqdm(url2docid.items(), desc=f"Compressing the {lang} docid_map"):
            url2docid[url] = list(docids.values()) if len(docids) == 1 else docids  # {code_tokens: docid} -> [docid]
        return url2docid

    def _get_n_docid(self):
        """ calculate the number of document ids contained in the nested docid map """
        lens = [len(docs) for url, docs in self._docid_map.items()]
        return sum(lens)

    def get_docid(self, url, code_tokens):
        """ retrieve the doc id according to the doc dict """
        docids = self.docid_map[url]
        return docids[0] if len(docids) == 1 else docids[code_tokens]
