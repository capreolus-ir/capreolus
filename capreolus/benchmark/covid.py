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
    qrel_url_v1 = "https://ir.nist.gov/covidSubmit/data/qrels-rnd%d.txt"
    qrel_url_v2 = "https://ir.nist.gov/covidSubmit/data/qrels-covid_d%d_j0.5-%d.txt"
    lastest_round = 5
    query_type = "title"

    config_spec = [ConfigOption("udelqexpand", False), ConfigOption("useprevqrels", True)]

    def build(self):
        if self.collection.config["round"] == self.lastest_round and not self.config["useprevqrels"]:
            logger.warning(f"No evaluation can be done for the lastest round without using previous qrels")

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

        rnd_i, useprevqrels = self.collection.config["round"], self.config["useprevqrels"]
        if rnd_i > self.lastest_round:
            raise ValueError(f"round {rnd_i} is unavailable")

        logger.info(f"Preparing files for covid round-{rnd_i}")

        # topic file
        topic_url = self.topic_url % rnd_i
        tmp_dir = self.get_cache_path() / "tmp"
        topic_tmp = tmp_dir / f"topic.round.{rnd_i}.xml"
        if not os.path.exists(topic_tmp):
            tmp_dir.mkdir(exist_ok=True, parents=True)
            download_file(topic_url, topic_tmp)
        all_qids = self.xml2trectopic(topic_tmp)  # will update self.topic_file
        labeled_qids = set()

        # put qrels from previous round into qrel_file if using previous judgement, else into qrel_ignore_file
        prev_qrel_urls = (
            [self.qrel_url_v1 % i for i in range(1, rnd_i)] if rnd_i <= 3 else [self.qrel_url_v2 % (rnd_i - 1, rnd_i - 1)]
        )  # qrels before current run
        # if rnd_i < 4:
        #     prev_qrel_urls = [self.qrel_url_v1 % i if rnd_i != 4 else self.qrel_url_v2 % (rnd_i-1, rnd_i-1) for i in range(1, rnd_i)]
        # elif rnd_i == 5:
        #     prev_qrel_urls = [self.qrel_url_v2 % (4, 4), self.qrel_url_v1 % 5]
        qrel_fn = open(self.qrel_file, "w") if useprevqrels else open(self.qrel_ignore, "w")

        for qrel_url in prev_qrel_urls:
            qrel_tmp = tmp_dir / qrel_url.split("/")[-1]
            if not os.path.exists(qrel_tmp):
                download_file(qrel_url, qrel_tmp)
            with open(qrel_tmp) as f:
                for line in f:
                    labeled_qids.add(line.strip().split()[0])
                    qrel_fn.write(line)
        qrel_fn.close()

        if useprevqrels:  # for rounds without available qrels
            f = open(self.qrel_ignore, "w")  # no qrels to remove after search
            f.close()
        # if not use previous qrels: use judgement in current round to evaluate
        elif rnd_i == self.lastest_round:
            logger.warn(f"No evaluation qrel is available for current round {rnd_i}")
            f = open(self.qrel_file, "w")
            f.close()
        elif rnd_i >= 3:  # special case since document id changes a lot from rnd 2 -> 3, or rnd 3 -> 4
            self.prep_backward_compatible_qrels(tmp_dir, self.qrel_ignore, self.qrel_file)  # write results to self.qrel
        else:  # not useprevqrels and rnd_i == 2,
            qrel_tmp = tmp_dir / f"qrel-{rnd_i}"
            if not os.path.exists(qrel_tmp):
                qrel_url = self.qrel_url_v1 % rnd_i if rnd_i != 4 else self.qrel_url_v2 % (4, 4)
                download_file(qrel_url, qrel_tmp)
            with open(qrel_tmp) as fin, open(self.qrel_file, "w") as fout:
                for line in fin:
                    fout.write(line)

        folds = {"s1": {"train_qids": list(labeled_qids), "predict": {"dev": list(labeled_qids), "test": all_qids}}}
        json.dump(folds, open(self.fold_file, "w"))

    def prep_backward_compatible_qrels(self, tmp_dir, prev_qrels_fn, tgt_qrel_fn):
        """
        Prepare qrels file for round 3 adaptable to previous rounds:
            convert the new docids in qrels-covid_d3_j0.5-3.txt back to its old id
            remove judgement existed in round1 and round2

        Warning: this function should not be used when search / training is done on collection
        released since round 4, where docids are already updated

        :param tmp_dir: pathlib.Path object, sthe directory to store downloaded files
        :param prev_qrels_fn: qrels file which store the qrels from previous rounds (round 1 and round 2)
        :param tgt_qrel_fn: qrels file path where to store the processed round 3 qrels file
        """
        DOCID2URL = {
            "rnd-3": "https://ir.nist.gov/covidSubmit/data/changedIds-May19.csv",
            "rnd-4": "https://ir.nist.gov/covidSubmit/data/changedIds-Jun19.csv",
        }
        rnd_i = self.collection.config["round"]
        assert rnd_i in [3, 4, 5]
        # assert self.collection.config["round"] == 3

        # donwload files
        qrel_url = f"https://ir.nist.gov/covidSubmit/data/qrels-covid_d{rnd_i}_j0.5-{rnd_i}.txt"
        docid_map_url = DOCID2URL[f"rnd-{rnd_i}"]
        qrel_tmp, docid_map_tmp = tmp_dir / f"qrel-{rnd_i}.before-convert", tmp_dir / f"round{rnd_i-1}-{rnd_i}.docid.map"
        if not qrel_tmp.exists():
            download_file(qrel_url, qrel_tmp)
        if not docid_map_tmp.exists():
            download_file(docid_map_url, docid_map_tmp)

        with open(docid_map_tmp) as f:  # docids to revert in current qrels file
            new2old = {line.split(",")[1]: line.split(",")[0] for line in f}  # each line: old_docid, new_docid, type

        with open(prev_qrels_fn) as f:  # qrels to exclude from current qrels file
            prev_qrels = [line for line in f]

        with open(qrel_tmp) as fin, open(tgt_qrel_fn, "w") as fout:
            for line in fin:
                qid, tag, docid, label = line.strip().split()
                docid = new2old.get(docid, docid)
                line = f"{qid} {tag}  {docid} {label}\n"  # covid qrel files have two whitespace between tag and docid
                if line not in prev_qrels:
                    fout.write(line)

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

        tmp_dir = self.get_cache_path() / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
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
