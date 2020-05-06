import os
import json
import glob
import gzip
import math
import subprocess
from threading import Thread
from multiprocessing import get_context
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import numpy as np

from pyserini.search import pysearch
from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS, PACKAGE_PATH
from capreolus.utils.common import Anserini
from capreolus.utils.trec import load_trec_topics
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
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"
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
        k1max = 1.0  # maximum k1 value to include in grid search
        bmax = 1.0  # maximum b value to include in grid search
        k1min = 0.1  # minimum k1 value to include in grid search
        bmin = 0.1  # minimum b value to include in grid search
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        bs = np.around(np.arange(self.cfg["bmin"], self.cfg["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(self.cfg["k1min"], self.cfg["k1max"] + 0.1, 0.1), 1)

        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = self.cfg["hits"]
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"

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
            + " ".join(f"-bm25.{k} {paras[k]}" for k in ["k1", "b"])
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


class BM25Reranker(Searcher):
    name = "BM25_reranker"
    dependencies = {
        "index": Dependency(module="index", name="anserini_tf"),
        "searcher": Dependency(module="searcher", name="csn_distractors"),
    }

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9
        hits = 1000

    def _calc_bm25(self, query, docid):
        return np.nansum([self["index"].get_bm25_weight(qterm, docid) for qterm in query.split()])

    def __calc_bm25(self, query, docid):
        k1, b = self.cfg["k1"], self.cfg["b"]
        avg_doc_len = self["index"].get_avglen()
        doclen = self["index"].get_doclen(docid)
        if doclen == -1:
            return -math.inf

        def term_bm25(term):
            tf = self["index"].get_tf(term, docid)
            if tf == math.nan:
                return math.nan

            idf = self["index"].get_idf(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doclen / avg_doc_len)
            return idf * numerator / denominator

        bm25_per_qterm = [term_bm25(qterm) for qterm in query.split()]
        return np.nansum(bm25_per_qterm)

    # def calc_bm25(self, qid, query, docids):  # for multiprocess setup
    #     if not docids:
    #         return {qid: {}}
    #
    #     bm25 = {docid: self._calc_bm25(query, docid) for docid in docids}
    #     bm25 = sorted(bm25.items(), key=lambda k_v: k_v[1], reverse=True)
    #     bm25 = {docid: bm25 for i, (docid, bm25) in enumerate(bm25) if i < self.cfg["hits"]}
    #     return qid, bm25

    def calc_bm25(self, query, docids):
        # bm25 = {docid: self._calc_bm25(query, docid) for docid in docids}
        bm25 = {docid: self.__calc_bm25(query, docid) for docid in docids}
        bm25 = sorted(bm25.items(), key=lambda k_v: k_v[1], reverse=True)
        bm25 = {docid: bm25 for i, (docid, bm25) in enumerate(bm25) if i < self.cfg["hits"]}
        return bm25

    def query_from_file(self, topicsfn, output_path, runs=None):
        """ only perform bm25 on the docs in runs """
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"done file for {self.name} already exists, skip search")
            return output_path

        topics = load_trec_topics(topicsfn)["title"]
        # qid_query_docids = [(qid, query, runs.get(qid, {})) for qid, query in topics.items()]
        # with get_context("spawn").Pool(10) as p:
        #     bm25_lists = p.starmap(self.calc_bm25, qid_query_docids)
        # assert len(bm25_lists) == len(qid_query_docids)
        # bm25runs = {qid: bm25 for qid, bm25 in bm25_lists}

        bm25runs = {}
        docnos = self["index"].open()
        docnos = self["index"]["collection"].get_docnos()

        def bm25(qid_queries):
            for qid, query in qid_queries:
                docids = runs.get(qid, None) if runs else docnos
                bm25runs[qid] = self.calc_bm25(query, docids) if docids else {}

        topics = list(topics.items())
        n_thread, threads = 5, []
        chunk_size = (len(topics) // n_thread) + 1
        for i in range(n_thread):
            start, end = i*chunk_size, (i+1)*chunk_size
            t = Thread(target=bm25, args=[topics[start:end]])
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # for qid, query in tqdm(topics.items(), desc=f"Calculating bm25"):
        #     docids = runs.get(qid, None)
        #     bm25runs[qid] = self.calc_bm25(query, docids) if docids else {}

        os.makedirs(output_path, exist_ok=True)
        print(f"runs: {len(bm25runs)}")
        self.write_trec_run(bm25runs, os.path.join(output_path, "searcher"))

        with open(donefn, "wt") as donef:
            print("done", file=donef)

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


class CodeSearchDistractor(Searcher):
    """ Providing the 999 distractor documents """

    name = "csn_distractors"
    dependencies = {"benchmark": Dependency(module="benchmark", name="codesearchnet_corpus")}

    def query_from_file(self, topicsfn, output_path):
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"done file for {self.name} already exists, skip search")
            return str(output_path)

        benchmark = self["benchmark"]
        lang = benchmark.cfg["lang"]

        csn_rawdata_dir, _ = benchmark.download_raw_data()
        csn_lang_dir = os.path.join(csn_rawdata_dir, lang, "final", "jsonl")

        runs = defaultdict(dict)
        # for set_name in ["train", "valid", "test"]:
        for set_name in ["valid", "test"]:
            csn_lang_path = os.path.join(csn_lang_dir, set_name)

            objs = []
            for fn in glob.glob(os.path.join(csn_lang_path, "*.jsonl.gz")):
                with gzip.open(fn, "rb") as f:
                    lines = f.readlines()
                    for line in tqdm(lines, desc=f"Processing set {set_name} {os.path.basename(fn)}"):
                        objs.append(json.loads(line))

                        if len(objs) == 1000:  # 1 ground truth and 999 distractor docs
                            for obj1 in objs:
                                qid = benchmark.get_qid(obj1["docstring_tokens"], parse=True)
                                gt_docid = benchmark.get_docid(obj1["url"], obj1["code_tokens"], parse=True)
                                all_docs = []

                                for rank, obj2 in enumerate(objs):
                                    docid = benchmark.get_docid(obj2["url"], obj2["code_tokens"], parse=True)
                                    all_docs.append(docid)
                                    runs[qid][docid] = 1.0 / (rank + 1)
                                assert gt_docid in all_docs
                            objs = []  # reset

        os.makedirs(output_path, exist_ok=True)
        print(f"runs: {len(runs)}")
        self.write_trec_run(runs, os.path.join(output_path, "searcher"))

        with open(donefn, "wt") as donef:
            print("done", file=donef)
        return str(output_path)

    def query(self, *args, **kwargs):
        raise NotImplementedError("this searcher uses a static run file, so it cannot handle new queries")
