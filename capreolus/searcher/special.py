import os
from collections import defaultdict
from tqdm import tqdm

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_trec_topics, topic_to_trectxt

from . import Searcher
from .anserini import BM25

logger = get_logger(__name__)


class MsmarcoPsgSearcherMixin:
    @staticmethod
    def convert_to_trec_runs(msmarco_top1k_fn, train):
        runs = defaultdict(dict)
        with open(msmarco_top1k_fn, "r", encoding="utf-8") as f:
            for line in f:
                if train:
                    qid, pos_pid, neg_pid = line.strip().split("\t")
                    runs[qid][pos_pid] = len(runs.get(qid, {}))
                    runs[qid][neg_pid] = len(runs.get(qid, {}))
                else:
                    qid, pid, _, _ = line.strip().split("\t")
                    runs[qid][pid] = len(runs.get(qid))
        return runs

    @staticmethod
    def get_fn_from_url(url):
        return url.split("/")[-1].replace(".gz", "").replace(".tar", "")

    def download_and_prepare_train_set(self, tmp_dir):
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz"
        extract_file_name = self.get_fn_from_url(url)
        extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
        runs = self.convert_to_trec_runs(extract_dir / extract_file_name, train=True)
        return runs


@Searcher.register
class MsmarcoPsg(Searcher, MsmarcoPsgSearcherMixin):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]

    def _query_from_file(self, topicsfn, output_path, cfg):
        """ only query results in dev and test set are saved """
        final_runfn = output_path / "searcher"
        final_donefn = output_path / "done"
        if os.path.exists(final_donefn):
            return output_path

        tmp_dir = self.get_cache_path() / "tmp"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        output_path.mkdir(exist_ok=True, parents=True)

        # train
        train_run = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        self.write_trec_run(preds=train_run, outfn=final_runfn, mode="wt")

        # dev and test
        dev_test_urls = [
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz",
        ]
        runs = {}
        for url in dev_test_urls:
            extract_file_name = self.get_fn_from_url(url)
            extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
            runs.update(self.convert_to_trec_runs(extract_dir / extract_file_name, train=False))
        self.write_trec_run(preds=runs, outfn=final_runfn, mode="a")

        with open(final_donefn, "wt") as f:
            print("done", file=f)
        return output_path


@Searcher.register
class MsmarcoPsgBm25(BM25, MsmarcoPsgSearcherMixin):
    module_name = "msmarcopsgbm25"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="msmarcopsg"),
        Dependency(key="index", module="index", name="anserini"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        final_runfn = os.path.join(output_path, "searcher")
        final_donefn = os.path.join(output_path, "done")
        if os.path.exists(final_donefn):
            return output_path

        output_path.mkdir(exist_ok=True, parents=True)
        tmp_dir = self.get_cache_path() / "tmp"
        tmp_topicsfn = tmp_dir / os.path.basename(topicsfn)
        tmp_output_dir = tmp_dir / "BM25_results"
        tmp_output_dir.mkdir(exist_ok=True, parents=True)

        train_runs = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        with open(tmp_topicsfn, "wt") as f:
            for qid, title in tqdm(load_trec_topics(topicsfn)["title"].items()):
                if qid not in self.benchmark.folds["s1"]["train_qids"]:
                    f.write(topic_to_trectxt(qid, title))
        super()._query_from_file(topicsfn=tmp_topicsfn, output_path=tmp_output_dir, config=config)
        dev_test_runfile = tmp_output_dir / "searcher"
        assert os.path.exists(dev_test_runfile)

        # write train and dev, test runs into final searcher file
        Searcher.write_trec_run(train_runs, final_runfn)
        with open(dev_test_runfile) as fin, open(final_runfn, "a") as fout:
            for line in fin:
                fout.write(line)

        with open(final_donefn, "w") as f:
            f.write("done")
        return output_path
