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
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the offical runfile for the development and the test set.
    """

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
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Conduct configurable BM25 search on the development and the test set.
    """

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


# todo: make this another type of "Module" (e.g. DPR Module)
@Searcher.register
class StaticTctColBertDev(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the runfile pre-prepared using TCT-ColBERT (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf)
    """

    module_name = "static_tct_colbert"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, cfg):
        outfn = output_path / "static.run"
        if outfn.exists():
            return outfn

        tmp_dir = self.get_cache_path() / "tmp"
        output_path.mkdir(exist_ok=True, parents=True)

        # train
        train_runs = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        self.write_trec_run(preds=train_runs, outfn=outfn, mode="wt")
        logger.info(f"prepared runs from train set")

        # dev
        tmp_dev = tmp_dir / "tct_colbert_v1_wo_neg.tsv"
        if not tmp_dev.exists():
            tmp_dir.mkdir(exist_ok=True, parents=True)
            url = "http://drive.google.com/uc?id=1jOVL3DIya6qDiwM_Dnqc81FT5ZB43csP"
            gdown.download(url, tmp_dev.as_posix(), quiet=False)

        assert tmp_dev.exists()
        with open(tmp_dev, "rt") as f, open(outfn, "at") as fout:
            for line in f:
                qid, docid, rank, score = line.strip().split("\t")
                fout.write(f"{qid} Q0 {docid} {rank} {score} tct_colbert\n")
        return outfn


@Searcher.register
class Tct2Marco(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the runfile pre-prepared using TCT-ColBERT (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf)
    """

    module_name = "msptop200"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption(
            "firststage",
            "tct",
            "Options: tct, bm25, tct>bm25, bm25>tct. where config before > stands for training set source, and that after > stands for dev and test source.",
        )
    ]

    def get_train_url(self):
        train_first_stage = self.config["firststage"].split(">")[0]

        url_template = "https://drive.google.com/uc?id="
        assert train_first_stage in {"bm25", "tct"}
        file_id = "10VjzcDUtZwJWoWUlVnjtyI4j5K6c-882" if train_first_stage == "tct" else "1ZgrxqdbV3-YbF9PnOVtSIx04RqG-YOMW"
        return url_template + file_id

    def get_dev_url(self):
        dev_first_stage = self.config["firststage"]
        if ">" in dev_first_stage:
            dev_first_stage = dev_first_stage.split(">")[1]

        url_template = "https://drive.google.com/uc?id="
        assert dev_first_stage in {"bm25", "tct"}
        file_id = "1WBUashNhtJKNsKYBzeR4IxcMzbjqiqg6" if dev_first_stage == "tct" else "1PWuDcr8c4EIB-mxdFY7-KkTezJ7aN0Fq"
        return url_template + file_id

    def get_test_url(self):
        dev_first_stage = self.config["firststage"]
        if ">" in dev_first_stage:
            dev_first_stage = dev_first_stage.split(">")[1]

        url_template = "https://drive.google.com/uc?id="
        assert dev_first_stage in {"tct"}, "Only support inference on tct test set for now"
        file_id = "1U4DBP_3HBXC8EJNbI_wFUVoZnt7FiPbe"
        return url_template + file_id

    def _query_from_file(self, topicsfn, output_path, cfg):
        outfn = output_path / "static.run"
        done_fn = output_path / "done"

        if done_fn.exists():
            assert outfn.exists()
            return outfn

        tmp_dir = self.get_cache_path() / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        output_path.mkdir(exist_ok=True, parents=True)

        tag = self.config["firststage"]
        fout = open(outfn, "wt")

        url_lists = [self.get_train_url(), self.get_dev_url()]
        if tag == "tct":
            url_lists.append(self.get_test_url())

        for set_name, url in zip(["train", "dev", "test"], url_lists):
            if set_name == "test":
                assert tag == "tct"

            # basename = self.get_fn_from_url(url)
            basename = f"{tag}-{set_name}"
            tmp_fn = tmp_dir / basename

            # download the file
            if not os.path.exists(tmp_fn):
                gdown.download(url, tmp_fn.as_posix(), quiet=False)

            # convert into trec and combine
            with open(tmp_fn, "rt") as f:
                for line in f:
                    try:
                        qid, docid, rank = line.strip().split()
                    except:
                        raise ValueError("This line cannot be parsed:" + line)

                    score = 1000 - int(rank)
                    fout.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

        with open(done_fn, "wt") as f:
            print("done", file=f)

        return outfn
