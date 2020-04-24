import json
import os
import gzip
import pickle

from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.trec import load_qrels, load_trec_topics, topic_to_trectxt
from capreolus.utils.common import download_file, hash_file


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


class CodeSearchNet(Benchmark):
    name = "codesearchnet"
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    query_type = "title"

    file_fn = PACKAGE_PATH / "data" / "csn"

    qrel_dir = file_fn / "qrels"
    topic_dir = file_fn / "topics"
    fold_dir = file_fn / "folds"

    docstring2qid_dir = file_fn / "docstring2qid"
    url2docid_dir = file_fn / "url2docid"

    @staticmethod
    def config():
        lang = "ruby"  # which language dataset under CodeSearchNet

    def __init__(self, cfg):
        super().__init__(cfg)
        lang = cfg["lang"]

        self.qid_map_file = self.docstring2qid_dir / f"{lang}.json"
        self.docid_map_file = self.url2docid_dir / f"{self.cfg['lang']}.json"

        self.qrel_file = self.qrel_dir / f"{self.cfg['lang']}.txt"
        self.topic_file = self.topic_dir / f"{self.cfg['lang']}.txt"
        self.fold_file = self.fold_dir / f"{self.cfg['lang']}.json"

        for file in [var for var in vars(self) if var.endswith("file")]:
            eval(f"self.{file}").parent.mkdir(exist_ok=True, parents=True)  # TODO: is eval a good coding habit?

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

            self._docid_map = json.load(open(self.docid_map_file, "r") )
        return self._docid_map

    @property
    def qrels(self):
        if not hasattr(self, "_qrels"):
            if not self.qrel_file.exists():
                self.download_if_missing()

            self._qrels = load_qrels(self.qrel_file)
        return self._qrels

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            if not self.topic_file.exists():
                self.download_if_missing()

            self._topics = load_trec_topics(self.topic_file)
        return self._topics

    @property
    def folds(self):
        if not hasattr(self, "_folds"):
            if not self.fold_file.exists():
                self.download_if_missing()

            self._folds = json.load(open(self.fold_file, "rt"))
        return self._folds

    def download_if_missing(self):
        lang = self.cfg["lang"]

        tmp_dir = Path("/tmp")
        zip_fn = tmp_dir / f"{lang}.zip"
        if not zip_fn.exists():
            download_file(f"{self.url}/{lang}.zip", zip_fn)

        with ZipFile(zip_fn, "r") as zipobj:
            zipobj.extractall(tmp_dir)

        # prepare docid-url mapping from dedup.pkl
        pkl_fn = tmp_dir / f"{lang}_dedupe_definitions_v2.pkl"
        with open(pkl_fn, "rb") as f:
            doc_objs = pickle.load(f)
        self._docid_map = self._prep_url2docid(doc_objs)
        assert self._get_n_docid() == len(doc_objs)
        with open(self.docid_map_file, "w") as f:
            json.dump(self._docid_map, f)

        # prepare folds, qrels, topics, docstring2qid  # TODO: shall we add negative samples?
        qrels, docstring2qid = defaultdict(dict), {}
        qids = {s: [] for s in ["train", "valid", "test"]}

        topic_file = open(self.topic_file, "w", encoding="utf-8")
        qrel_file = open(self.qrel_file , "w", encoding="utf-8")

        for set_name in qids:
            set_path = tmp_dir / lang / "final" / "jsonl" / set_name
            for fn in sorted(set_path.glob("*.jsonl.gz")):
                f = gzip.open(fn, "rb")
                for doc in f:
                    doc = json.loads(doc)
                    docstring, code = " ".join(doc["docstring_tokens"]), " ".join(doc["code_tokens"])
                    docid = self._get_docid(doc)

                    if docstring in docstring2qid:
                        qid = docstring2qid[docstring]
                    else:
                        qid = f"{lang}-DOCSTRING-{len(docstring2qid)}"
                        qids[set_name].append(qid)      # for fold.json
                        docstring2qid[docstring] = qid  # for docstring2qid.json
                        topic_file.write(topic_to_trectxt(qid, docstring))  # write to topic.txt

                    qrel_file.write(f"{qid} Q0 {docid} 1\n")  # write to qrels.txt
        topic_file.close()
        qrel_file.close()

        # write to fold.json, docstring2qid
        with open(self.qid_map_file, "w") as f:
            json.dump(docstring2qid, f)
        with open(self.fold_file, "w") as f:
            json.dump({"s1": {
                "train": qids["train"],
                "predict": {"dev": qids["valid"], "test": qids["test"]}
            }}, f)

    def _prep_url2docid(self, doc_objs):
        """
        construct a nested dict to map each doc into a unique docid
        the dict structure: {url: {" ".join(code_tokens): docid, ...}}

        For all the lanugage datasets but js and php in csn, url uniquely maps to a code_tokens, which makes a nested mapping is necessary.

        :param doc_objs:
        :return:
        """
        # TODO: any way to avoid the twice traversal of all url and make the return dict structure consistent

        url2docid = defaultdict(dict)
        for i, doc in enumerate(doc_objs):
            lang = self.cfg["lang"]
            url, code_tokens = doc["url"], " ".join(doc["function_tokens"])
            url2docid[url][code_tokens] = f"{lang}-FUNCTION-{i}"

        # remove the code_tokens for the unique url-docid mapping
        for url in url2docid:
            if len(url2docid[url]) == 1:
                url2docid[url] = list(url2docid[url].values())  # {code_tokens: docid} -> [docid]
        return url2docid

    def _get_n_docid(self):
        lens = [len(docs) for url, docs in self._docid_map.items()]
        return sum(lens)

    def _get_docid(self, doc):
        """ retrieve the doc id according to the doc dict """
        url, code_tokens = doc["url"], " ".join(doc["code_tokens"])
        docids = self._docid_map[url]
        return docids[0] if len(docids) == 1 else docids[code_tokens]
