import os
import gdown
from pathlib import Path
from collections import defaultdict

from capreolus import ConfigOption, Dependency
from capreolus.utils.loginit import get_logger

from . import Searcher
from .anserini import BM25

logger = get_logger(__name__)

SUPPORTED_TRIPLE_FILE = ["small", "large.v1", "large.v2"]


# def get_file_line_number(fn):
#     return int(os.popen(f"wc -l {fn}").readline().split()[0])

class MsmarcoPsgSearcherMixin:
    @staticmethod
    def convert_to_trec_runs(msmarco_top1k_fn, style="eval"):
        logger.info(f"Converting file {msmarco_top1k_fn} (with style {style}) into trec format")
        runs = defaultdict(dict)
        with open(msmarco_top1k_fn, "r", encoding="utf-8") as f:
            for line in f:
                if style == "triple":
                    qid, pos_pid, neg_pid = line.strip().split("\t")
                    runs[qid][pos_pid] = len(runs.get(qid, {}))
                    runs[qid][neg_pid] = len(runs.get(qid, {}))
                elif style == "eval":
                    qid, pid, _, _ = line.strip().split("\t")
                    runs[qid][pid] = len(runs.get(qid, []))
                else:
                    raise ValueError(f"Unexpected style {style}, should be either 'triple' or 'eval'")
        return runs

    @staticmethod
    def get_fn_from_url(url):
        return url.split("/")[-1].replace(".gz", "").replace(".tar", "")

    def get_url(self):
        tripleversion = self.config["tripleversion"]
        if tripleversion == "large.v1":
            return "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz"

        if tripleversion == "large.v2":
            return "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"

        if tripleversion == "small":
            return "https://drive.google.com/uc?id=1LCQ-85fx61_5gQgljyok8olf6GadZUeP"

        raise ValueError("Unknown version for triplet large" % self.config["tripleversion"])

    def download_and_prepare_train_set(self, tmp_dir):
        tmp_dir.mkdir(exist_ok=True, parents=True)
        triple_version = self.config["tripleversion"]

        url = self.get_url()
        if triple_version.startswith("large"):
            extract_file_name = self.get_fn_from_url(url)
            extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
            triple_fn = extract_dir / extract_file_name
        elif triple_version == "small":
            triple_fn = tmp_dir / "triples.train.small.idversion.tsv"
            if not os.path.exists(triple_fn):
                gdown.download(url, triple_fn.as_posix(), quiet=False)
        else:
            raise ValueError(f"Unknown version for triplet: {triple_version}")

        return self.convert_to_trec_runs(triple_fn, style="triple")


@Searcher.register
class MsmarcoPsg(Searcher, MsmarcoPsgSearcherMixin):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, cfg):
        """only query results in dev and test set are saved"""
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
            runs.update(self.convert_to_trec_runs(extract_dir / extract_file_name, style="eval"))
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
    config_spec = BM25.config_spec + [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
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
        if not os.path.exists(tmp_topicsfn):
            with open(tmp_topicsfn, "wt") as fout:
                with open(topicsfn) as f:
                    for line in f:
                        qid, title = line.strip().split("\t")
                        if qid not in self.benchmark.folds["s1"]["train_qids"]:
                            fout.write(line)

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


class MSMARCO_V2_SearcherMixin:
    def get_train_runfile(self):
        raise NotImplementedError

    def combine_train_and_dev_runfile(self, dev_test_runfile, final_runfile, final_donefn):
        train_runfile = self.get_train_runfile()
        assert os.path.exists(dev_test_runfile)

        # write train and dev, test runs into final searcher file
        with open(final_runfile, "w") as fout:
            with open(train_runfile) as fin:
                for line in fin:
                    fout.write(line)

            with open(dev_test_runfile) as fin:
                for line in fin:
                    fout.write(line)

        with open(final_donefn, "w") as f:
            f.write("done")


@Searcher.register
class MSMARCO_V2_Bm25(BM25, MSMARCO_V2_SearcherMixin):
    module_name = "msv2bm25"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="msdoc_v2"),
        Dependency(key="index", module="index", name="anserini"),
    ]
    config_spec = BM25.config_spec

    def get_train_runfile(self):
        basename = f"{self.benchmark.dataset_type}v2_train_top100.txt" 
        return self.benchmark.data_dir / basename

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
        print(tmp_output_dir)

        # run bm25 on dev set
        if not os.path.exists(tmp_topicsfn):
            with open(tmp_topicsfn, "wt") as f:
                # i, n_expected = 0, 326748 if self.benchmark.config["type"] == "doc" else 281047
                logger.info("Preparing tmp topic file for bm25")
                for line in open(topicsfn): 
                    qid, title = line.strip().split("\t")
                    if qid not in self.benchmark.folds["s1"]["train_qids"]:
                        f.write(f"{qid}\t{title}\n")
                    # i += 1
                # assert i == n_expected

        super()._query_from_file(topicsfn=tmp_topicsfn, output_path=tmp_output_dir, config=config)
        self.combine_train_and_dev_runfile(
            tmp_output_dir / "searcher",
            final_runfn,
            final_donefn,
        )

        return output_path
