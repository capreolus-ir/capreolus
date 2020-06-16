from profane import import_all_modules

import json
import re
import os
import gzip
import pickle

from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
from profane import ModuleBase, Dependency, ConfigOption, constants

from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels, load_trec_topics, topic_to_trectxt
from capreolus.utils.common import download_file, remove_newline, get_udel_query_expander

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


class Benchmark(ModuleBase):
    """the module base class"""

    module_type = "benchmark"
    qrel_file = None
    topic_file = None
    fold_file = None
    query_type = None
    # documents with a relevance label >= relevance_level will be considered relevant
    # corresponds to trec_eval's --level_for_rel (and passed to pytrec_eval as relevance_level)
    relevance_level = 1

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


@Benchmark.register
class DummyBenchmark(Benchmark):
    module_name = "dummy"
    dependencies = [Dependency(key="collection", module="collection", name="dummy")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.dummy.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.dummy.txt"
    fold_file = PACKAGE_PATH / "data" / "dummy_folds.json"
    query_type = "title"


@Benchmark.register
class WSDM20Demo(Benchmark):
    """ Robust04 benchmark equivalent to robust04.yang19 """

    module_name = "wsdm20demo"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


@Benchmark.register
class Robust04Yang19(Benchmark):
    """Robust04 benchmark using the folds from Yang et al. [1]

    [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. 2019. Critically Examining the "Neural Hype": Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    module_name = "robust04.yang19"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


@Benchmark.register
class NF(Benchmark):
    """ A Full-Text Learning to Rank Dataset for Medical Information Retrieval [1]

    [1] Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler. A Full-Text Learning to Rank Dataset for Medical Information Retrieval Proceedings of the 38th European Conference on Information Retrieval (ECIR), Padova, Italy, 2016
    """

    module_name = "nf"
    dependencies = [Dependency(key="collection", module="collection", name="nf")]
    config_spec = [ConfigOption(key="fields", default_value="all", description="query fields included in topic file")]

    qrel_file = PACKAGE_PATH / "data" / "qrels.nf.txt"
    test_qrel_file = PACKAGE_PATH / "data" / "test.qrels.nf.txt"
    fold_file = PACKAGE_PATH / "data" / "nf.json"

    query_type = "title"

    def __init__(self, config, provide, share_dependency_objects):
        super().__init__(config, provide, share_dependency_objects)
        fields = self.config["fields"] + "-field"
        self.topic_file = PACKAGE_PATH / "data" / f"topics.nf.{fields}.txt"
        self.download_if_missing()

    def _transform_qid(self, raw):
        """ NFCorpus dataset specific, remove prefix in query id since anserini convert all qid to integer """
        return raw.replace("PLAIN-", "")

    def download_if_missing(self):
        if all([f.exists() for f in [self.topic_file, self.fold_file, self.qrel_file]]):
            return

        tmp_corpus_dir = self.collection.download_raw()
        topic_f = open(self.topic_file, "w", encoding="utf-8")
        qrel_f = open(self.qrel_file, "w", encoding="utf-8")
        test_qrel_f = open(self.test_qrel_file, "w", encoding="utf-8")

        set_names = ["train", "dev", "test"]
        folds = {s: set() for s in set_names}
        for set_name in set_names:
            with open(tmp_corpus_dir / f"{set_name}.2-1-0.qrel") as f:
                for line in f:
                    line = self._transform_qid(line)
                    qid = line.strip().split()[0]
                    folds[set_name].add(qid)
                    if set_name == "test":
                        test_qrel_f.write(line)
                    qrel_f.write(line)

            field2files, qids2topics = {}, {}
            if self.config["fields"] == "all":
                field2files["title"] = [tmp_corpus_dir / f"{set_name}.all.queries"]
            else:
                field2files["title"] = [
                    tmp_corpus_dir / f"{set_name}.{subname}.queries" for subname in ["titles", "vid-titles", "nontopic-titles"]
                ]
                field2files["desc"] = [tmp_corpus_dir / f"{set_name}.vid-desc.queries"]

            for field, files in field2files.items():
                qids2topics = self._align_queries(files, field, qids2topics)
            for qid, txts in qids2topics.items():
                topic_f.write(topic_to_trectxt(qid, txts.get("title", None), txts.get("desc", None)))

        json.dump(
            {"s1": {"train_qids": list(folds["train"]), "predict": {"dev": list(folds["dev"]), "test": list(folds["test"])}}},
            open(self.fold_file, "w"),
        )

        topic_f.close()
        qrel_f.close()
        test_qrel_f.close()
        logger.info(f"nf benchmark prepared")

    def _align_queries(self, files, field, qid2queries=None):
        if not qid2queries:
            qid2queries = {}
        for fn in files:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    qid, txt = line.strip().split("\t")
                    qid = self._transform_qid(qid)
                    txt = " ".join(re.sub("[^A-Za-z]", " ", txt).split()[:1020])
                    if qid not in qid2queries:
                        qid2queries[qid] = {field: txt}
                    else:
                        if field in qid2queries[qid]:
                            logger.warning(f"Overwriting title for query {qid}")
                        qid2queries[qid][field] = txt
        return qid2queries


@Benchmark.register
class ANTIQUE(Benchmark):
    """A Non-factoid Question Answering Benchmark from Hashemi et al. [1]

    [1] Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W. Bruce Croft. 2020. ANTIQUE: A non-factoid question answering benchmark. ECIR 2020.
    """

    module_name = "antique"
    dependencies = [Dependency(key="collection", module="collection", name="antique")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.antique.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.antique.txt"
    fold_file = PACKAGE_PATH / "data" / "antique.json"
    query_type = "title"
    relevance_level = 2


@Benchmark.register
class MSMarcoPassage(Benchmark):
    module_name = "msmarcopassage"
    dependencies = [Dependency(key="collection", module="collection", name="msmarco")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcopassage.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.msmarcopassage.txt"
    fold_file = PACKAGE_PATH / "data" / "msmarcopassage.folds.json"
    query_type = "title"


@Benchmark.register
class CodeSearchNetCorpus(Benchmark):
    module_name = "codesearchnet_corpus"
    dependencies = [Dependency(key="collection", module="collection", name="codesearchnet")]
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    query_type = "title"

    file_fn = PACKAGE_PATH / "data" / "csn_corpus"

    qrel_dir = file_fn / "qrels"
    topic_dir = file_fn / "topics"
    fold_dir = file_fn / "folds"

    qidmap_dir = file_fn / "qidmap"
    docidmap_dir = file_fn / "docidmap"

    config_spec = [ConfigOption("lang", "ruby", "CSN language dataset to use")]

    def build(self):
        lang = self.config["lang"]

        self.qid_map_file = self.qidmap_dir / f"{lang}.json"
        self.docid_map_file = self.docidmap_dir / f"{lang}.json"

        self.qrel_file = self.qrel_dir / f"{lang}.txt"
        self.topic_file = self.topic_dir / f"{lang}.txt"
        self.fold_file = self.fold_dir / f"{lang}.json"

        for file in [var for var in vars(self) if var.endswith("file")]:
            getattr(self, file).parent.mkdir(exist_ok=True, parents=True)

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

        lang = self.config["lang"]

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
                    logger.warning(
                        f"chunk query to first 1000 words otherwise TooManyClause would be triggered "
                        f"at lucene at search stage, "
                    )
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
        json.dump(
            {"s1": {"train_qids": qids["train"], "predict": {"dev": qids["valid"], "test": qids["test"]}}},
            open(self.fold_file, "w"),
        )

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
        lang = self.config["lang"]
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


@Benchmark.register
class CodeSearchNetChallenge(Benchmark):
    """
    CodeSearchNetChallenge can only be used for training but not for evaluation since qrels is not provided
    """

    module_name = "codesearchnet_challenge"
    dependencies = [Dependency(key="collection", module="collection", name="codesearchnet")]
    config_spec = [ConfigOption("lang", "ruby", "CSN language dataset to use")]

    url = "https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/queries.csv"
    query_type = "title"

    file_fn = PACKAGE_PATH / "data" / "csn_challenge"
    topic_file = file_fn / "topics.txt"
    qid_map_file = file_fn / "qidmap.json"

    def download_if_missing(self):
        """ download query.csv and prepare queryid - query mapping file """
        if self.topic_file.exists() and self.qid_map_file.exists():
            return

        tmp_dir = Path("/tmp")
        tmp_dir.mkdir(exist_ok=True, parents=True)
        self.file_fn.mkdir(exist_ok=True, parents=True)

        query_fn = tmp_dir / f"query.csv"
        if not query_fn.exists():
            download_file(self.url, query_fn)

        # prepare qid - query
        qid_map = {}
        topic_file = open(self.topic_file, "w", encoding="utf-8")
        query_file = open(query_fn)
        for qid, line in enumerate(query_file):
            if qid != 0:  # ignore the first line "query"
                topic_file.write(topic_to_trectxt(qid, line.strip()))
                qid_map[qid] = line
        topic_file.close()
        json.dump(qid_map, open(self.qid_map_file, "w"))


@Benchmark.register
class COVID(Benchmark):
    """ Ongoing TREC-COVID bechmark from https://ir.nist.gov/covidSubmit """

    module_name = "covid"
    dependencies = [Dependency(key="collection", module="collection", name="covid")]
    data_dir = PACKAGE_PATH / "data" / "covid"
    topic_url = "https://ir.nist.gov/covidSubmit/data/topics-rnd%d.xml"
    qrel_url = "https://ir.nist.gov/covidSubmit/data/qrels-rnd%d.txt"
    lastest_round = 3

    config_spec = [
        ConfigOption("round", 3, "TREC-COVID round to use"),
        ConfigOption("udelqexpand", False),
        ConfigOption("excludeknown", True),
    ]

    def build(self):
        if self.config["round"] == self.lastest_round and not self.config["excludeknown"]:
            logger.warning(f"No evaluation can be done for the lastest round in exclude-known mode")

        data_dir = self.get_cache_path() / "documents"
        data_dir.mkdir(exist_ok=True, parents=True)

        self.qrel_ignore = f"{data_dir}/ignore.qrel.txt"
        self.qrel_file = f"{data_dir}/qrel.txt"
        self.topic_file = f"{data_dir}/topic.txt"
        self.fold_file = f"{data_dir}/fold.json"

        self.download_if_missing()

    def download_if_missing(self):
        if all([os.path.exists(fn) for fn in [self.qrel_file, self.qrel_ignore, self.topic_file, self.fold_file]]):
            return

        rnd_i, excludeknown = self.config["round"], self.config["excludeknown"]
        if rnd_i > self.lastest_round:
            raise ValueError(f"round {rnd_i} is unavailable")

        logger.info(f"Preparing files for covid round-{rnd_i}")

        topic_url = self.topic_url % rnd_i
        qrel_ignore_urls = [self.qrel_url % i for i in range(1, rnd_i)]  # download all the qrels before current run

        # topic file
        tmp_dir = Path("/tmp")
        topic_tmp = tmp_dir / f"topic.round.{rnd_i}.xml"
        if not os.path.exists(topic_tmp):
            download_file(topic_url, topic_tmp)
        all_qids = self.xml2trectopic(topic_tmp)  # will update self.topic_file

        if excludeknown:
            qrel_fn = open(self.qrel_file, "w")
            for i, qrel_url in enumerate(qrel_ignore_urls):
                qrel_tmp = tmp_dir / f"qrel-{i+1}"  # round_id = (i + 1)
                if not os.path.exists(qrel_tmp):
                    download_file(qrel_url, qrel_tmp)
                with open(qrel_tmp) as f:
                    for line in f:
                        qrel_fn.write(line)
            qrel_fn.close()

            f = open(self.qrel_ignore, "w")  # empty ignore file
            f.close()
        else:
            qrel_fn = open(self.qrel_ignore, "w")
            for i, qrel_url in enumerate(qrel_ignore_urls):
                qrel_tmp = tmp_dir / f"qrel-{i+1}"  # round_id = (i + 1)
                if not os.path.exists(qrel_tmp):
                    download_file(qrel_url, qrel_tmp)
                with open(qrel_tmp) as f:
                    for line in f:
                        qrel_fn.write(line)
            qrel_fn.close()

            if rnd_i == self.lastest_round:
                f = open(self.qrel_file, "w")
                f.close()
            else:
                with open(tmp_dir / f"qrel-{rnd_i}") as fin, open(self.qrel_file, "w") as fout:
                    for line in fin:
                        fout.write(line)

        # folds: use all labeled query for train, valid, and use all of them for test set
        labeled_qids = list(load_qrels(self.qrel_ignore).keys())
        folds = {"s1": {"train_qids": labeled_qids, "predict": {"dev": labeled_qids, "test": all_qids}}}
        json.dump(folds, open(self.fold_file, "w"))

    def xml2trectopic(self, xmlfile):
        with open(xmlfile, "r") as f:
            topic = f.read()

        all_qids = []
        soup = BeautifulSoup(topic, "lxml")
        topics = soup.find_all("topic")
        expand_query = get_udel_query_expander()

        with open(self.topic_file, "w") as fout:
            for topic in topics:
                qid = topic["number"]
                title = topic.find_all("query")[0].text.strip()
                desc = topic.find_all("question")[0].text.strip()
                narr = topic.find_all("narrative")[0].text.strip()

                if self.config["udelqexpand"]:
                    title = expand_query(title, rm_sw=True)
                    desc = expand_query(desc, rm_sw=False)

                    title = title + " " + desc
                    desc = " "

                topic_line = topic_to_trectxt(qid, title, desc=desc, narr=narr)
                fout.write(topic_line)
                all_qids.append(qid)
        return all_qids


@Benchmark.register
class CovidQA(Benchmark):
    module_name = "covidqa"
    dependencies = [Dependency(key="collection", module="collection", name="covidqa")]
    url = "https://raw.githubusercontent.com/castorini/pygaggle/master/data/kaggle-lit-review-%s.json"
    available_versions = ["0.1", "0.2"]

    datadir = PACKAGE_PATH / "data" / "covidqa"

    config_spec = [ConfigOption("version", "0.1+0.2")]

    def build(self):
        os.makedirs(self.datadir, exist_ok=True)

        version = self.config["version"]
        self.qrel_file = self.datadir / f"qrels.v{version}.txt"
        self.topic_file = self.datadir / f"topics.v{version}.txt"
        self.fold_file = self.datadir / f"v{version}.json"  # HOW TO SPLIT THE FOLD HERE?

        self.download_if_missing()

    def download_if_missing(self):
        if all([os.path.exists(f) for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        tmp_dir = Path("/tmp")
        topic_f = open(self.topic_file, "w", encoding="utf-8")
        qrel_f = open(self.qrel_file, "w", encoding="utf-8")

        all_qids = []
        qid = 2001  # to distingsuish queries here from queries in TREC-covid
        versions = self.config["version"].split("+") if isinstance(self.config["version"], str) else str(self.config["version"])
        for v in versions:
            if v not in self.available_versions:
                vs = " ".join(self.available_versions)
                logger.warning(f"Invalid version {v}, should be one of {vs}")
                continue

            url = self.url % v
            target_fn = tmp_dir / f"covidqa-v{v}.json"
            if not os.path.exists(target_fn):
                download_file(url, target_fn)
            qa = json.load(open(target_fn))
            for subcate in qa["categories"]:
                name = subcate["name"]

                for qa in subcate["sub_categories"]:
                    nq_name, kq_name = qa["nq_name"], qa["kq_name"]
                    query_line = topic_to_trectxt(qid, kq_name, nq_name)  # kq_name == "query", nq_name == "question"
                    topic_f.write(query_line)
                    for ans in qa["answers"]:
                        docid = ans["id"]
                        qrel_f.write(f"{qid} Q0 {docid} 1\n")
                    all_qids.append(qid)
                    qid += 1

        json.dump({"s1": {"train_qids": all_qids, "predict": {"dev": all_qids, "test": all_qids}}}, open(self.fold_file, "w"))
        topic_f.close()
        qrel_f.close()


import_all_modules(__file__, __package__)
