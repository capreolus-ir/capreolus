import json
import os
from pathlib import Path

from bs4 import BeautifulSoup

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, get_udel_query_expander
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_qrels, topic_to_trectxt

from . import Benchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class COVID(Benchmark):
    """ Ongoing TREC-COVID bechmark from https://ir.nist.gov/covidSubmit that uses documents from CORD, the COVID-19 Open Research Dataset (https://www.semanticscholar.org/cord19). """

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
    dependencies = [Dependency(key="collection", module="collection", name="covid")]
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
