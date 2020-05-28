import os
import subprocess
from collections import defaultdict, OrderedDict

import numpy as np

from pyserini.search import pysearch
from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS, PACKAGE_PATH
from capreolus.utils.trec import load_qrels
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Searcher(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "searcher"

    @staticmethod
    def load_trec_run(fn):
        run = defaultdict(dict)
        with open(fn, "rt") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    qid, _, docid, rank, score, desc = line.split(" ")
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn):
        count = 0
        with open(outfn, "wt") as outf:
            for qid in sorted(preds):
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1


class AnseriniSearcherMixIn:
    """ MixIn for searchers that use Anserini's SearchCollection script """

    def _anserini_query_from_file(self, topicsfn, anserini_param_str, output_base_path):
        if not os.path.exists(topicsfn):
            raise IOError(f"could not find topics file: {topicsfn}")

        donefn = os.path.join(output_base_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"skipping Anserini SearchCollection call because path already exists: {donefn}")
            return

        # create index if it does not exist. the call returns immediately if the index does exist.
        self["index"].create_index()

        os.makedirs(output_base_path, exist_ok=True)
        output_path = os.path.join(output_base_path, "searcher")

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self['index'].cfg['stemmer']}"
        if self["index"].cfg["indexstops"]:
            indexopts += " -keepstopwords"

        index_path = self["index"].get_index_path()
        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader Trec -index {index_path} {indexopts} -topics {topicsfn} -output {output_path} -inmem -threads {MAX_THREADS} {anserini_param_str}"
        logger.info("Anserini writing runs to %s", output_path)
        logger.debug(cmd)

        app = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

        # Anserini output is verbose, so ignore DEBUG log lines and send other output through our logger
        for line in app.stdout:
            Anserini.filter_and_log_anserini_output(line, logger)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

        with open(donefn, "wt") as donef:
            print("done", file=donef)


class FilterMixin:
    def filter(self, run_dir, docs_to_remove=None, docs_to_keep=None, topn=None):
        if (not docs_to_keep) and (not docs_to_remove):
            raise

        run_dir = str(run_dir)
        outp_dir = f"{run_dir}_filtered"
        os.makedirs(outp_dir, exist_ok=True)
        for fn in os.listdir(run_dir):
            if fn == "done":
                continue
            # run_fn, outp_fn = os.path.join(run_dir, fn), os.path.join(outp_dir, fn)
            run_fn = os.path.join(run_dir, fn)
            self._filter(run_fn, docs_to_remove, docs_to_keep, topn)
        return outp_dir

    def _filter(self, runfile, docs_to_remove, docs_to_keep, topn):
        runs = Searcher.load_trec_run(runfile)

        # filtering
        if docs_to_remove:  # prioritize docs_to_remove
            if isinstance(docs_to_remove, list):
                docs_to_remove = {q: docs_to_remove for q in runs}
            runs = {q: {d: v for d, v in docs.items() if d not in docs_to_remove.get(q, [])} for q, docs in runs.items()}
        elif docs_to_keep:
            if isinstance(docs_to_keep, list):
                docs_to_keep = {q: docs_to_keep for q in runs}
            runs = {q: {d: v for d, v in docs.items() if d in docs_to_keep[q]} for q, docs in runs.items()}

        # keep the top k
        if not topn:
            Searcher.write_trec_run(runs, runfile)  # overwrite runfile

        queries = sorted(list(runs.keys()))
        for q in queries:
            docs = runs[q]
            if len(docs) <= topn:
                continue
            docs = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[:topn]
            runs[q] = {k: v for k, v in docs}
        Searcher.write_trec_run(runs, runfile)  # overwrite runfile


class BM25(Searcher, AnseriniSearcherMixIn):
    """ BM25 with fixed k1 and b. """

    name = "BM25"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        b = 0.4  # controls document length normalization
        k1 = 0.9  # controls term saturation
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        """
        Runs BM25 search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        bs = [self.cfg["b"]]
        k1s = [self.cfg["k1"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = self.cfg["hits"]
        anserini_param_str = f"-bm25 -b {bstr} -k1 {k1str} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(self.cfg["k1"], self.cfg["b"])

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25Grid(Searcher, AnseriniSearcherMixIn):
    """ BM25 with a grid search for k1 and b. Search is from 0.1 to bmax/k1max in 0.1 increments """

    name = "BM25Grid"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        k1max = 1.0  # maximum k1 value to include in grid search (starting at 0.1)
        bmax = 1.0  # maximum b value to include in grid search (starting at 0.1)
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        bs = np.around(np.arange(0.1, self.cfg["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(0.1, self.cfg["k1max"] + 0.1, 0.1), 1)
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = self.cfg["hits"]
        anserini_param_str = f"-bm25 -b {bstr} -k1 {k1str} -hits {hits}"

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query, b, k1):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25RM3(Searcher, AnseriniSearcherMixIn):

    name = "BM25RM3"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        k1 = BM25RM3.list2str([0.65, 0.70, 0.75])
        b = BM25RM3.list2str([0.60, 0.7])  # [0.60, 0.65, 0.7]
        fbTerms = BM25RM3.list2str([65, 70, 95, 100])
        fbDocs = BM25RM3.list2str([5, 10, 15])
        originalQueryWeight = BM25RM3.list2str([0.2, 0.25])
        hits = 1000

    @staticmethod
    def list2str(l):
        return "-".join(str(x) for x in l)

    def query_from_file(self, topicsfn, output_path):
        # paras = {k: self.list2str(self.cfg[k]) for k in ["k1", "b", "fbTerms", "fbDocs", "originalQueryWeight"]}
        paras = {k: " ".join(self.cfg[k].split("-")) for k in ["k1", "b", "fbTerms", "fbDocs", "originalQueryWeight"]}
        hits = str(self.cfg["hits"])

        anserini_param_str = (
            "-rm3 "
            + " ".join(f"-rm3.{k} {paras[k]}" for k in ["fbTerms", "fbDocs", "originalQueryWeight"])
            + " -bm25 "
            + " ".join(f"-{k} {paras[k]}" for k in ["k1", "b"])
            + f" -hits {hits}"
        )
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query, b, k1, fbterms, fbdocs, ow):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)
        searcher.set_rm3_reranker(fb_terms=fbterms, fb_docs=fbdocs, original_query_weight=ow)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25Filter(BM25, FilterMixin):
    name = "BM25Filter"

    @staticmethod
    def config():
        b = 0.4  # controls document length normalization
        k1 = 0.9  # controls term saturation
        hits = 1000
        topn = 1000

    def query_from_file(self, topicsfn, output_path):
        qrel_fn = "/home/xinyu1zhang/cikm/capreolus-covid/capreolus/data/covid/round2.ignore.qrel.txt"
        qrels = load_qrels(qrel_fn)
        docs_to_remove = {q: list(d.keys()) for q, d in qrels.items()}

        output_path = super().query_from_file(topicsfn, output_path)
        output_path = super().filter(output_path, docs_to_remove=docs_to_remove, topn=self.cfg["topn"])

        return output_path


class StaticBM25RM3Rob04Yang19(Searcher):
    """ Tuned BM25+RM3 run used by Yang et al. in [1]. This should be used only with a benchmark using the same folds and queries.

        [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and  the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    name = "bm25staticrob04yang19"

    def query_from_file(self, topicsfn, output_path):
        import shutil

        outfn = os.path.join(output_path, "static.run")
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(PACKAGE_PATH / "data" / "rob04_yang19_rm3.run", outfn)

        return output_path

    def query(self, *args, **kwargs):
        raise NotImplementedError("this searcher uses a static run file, so it cannot handle new queries")


class DirichletQL(Searcher, AnseriniSearcherMixIn):
    """ Dirichlet QL with a fixed mu """

    name = "DirichletQL"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        mu = 1000  # mu smoothing parameter
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        """
        Runs Dirichlet QL search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        mus = [self.cfg["mu"]]
        mustr = " ".join(str(x) for x in mus)
        hits = self.cfg["hits"]
        anserini_param_str = f"-qld -mu {mustr} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_lm_dirichlet_similarity(self.cfg["mu"])

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})
