from profane import import_all_modules


# import_all_modules(__file__, __package__)

import os
import math
import subprocess
from collections import defaultdict, OrderedDict

import numpy as np
from profane import ModuleBase, Dependency, ConfigOption, constants
from pyserini.search import pysearch

from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]


def list2str(l):
    return "-".join(str(x) for x in l)


class Searcher(ModuleBase):
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
            qids = sorted(preds.keys(), key=lambda k: int(k))
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    def _query_from_file(self, topicsfn, output_path, cfg):
        raise NotImplementedError()

    def query_from_file(self, topicsfn, output_path):
        return self._query_from_file(topicsfn, output_path, self.config)

    def query(self, query, **kwargs):
        """fire a search with parameters, using parameters in config as default"""
        config = {k: kwargs.get(k, self.config[k]) for k in self.config}

        cache_dir = self.get_cache_path()
        cache_dir.mkdir(exist_ok=True)
        topic_fn, runfile_dir = cache_dir / "topic.txt", cache_dir / "runfiles"
        with open(topic_fn, "w", encoding="utf-8") as f:
            f.write(topic_to_trectxt(1, query))

        self._query_from_file(topic_fn, runfile_dir, config)
        # TODO: in terms of grid search case, shall we import evaluator to find the best runfile?

        print(runfile_dir)
        runfile_fn = [f for f in os.listdir(runfile_dir) if f != "done"][0]  # assume only one file is generated
        runfile_fn = runfile_dir / runfile_fn

        runs = self.load_trec_run(runfile_fn)
        os.remove(runfile_fn)  # remove it in case the file accumulate
        os.remove(runfile_dir / "done")

        return OrderedDict(runs)


class AnseriniSearcherMixIn:
    """ MixIn for searchers that use Anserini's SearchCollection script """

    def _anserini_query_from_file(self, topicsfn, anserini_param_str, output_base_path, topicfield):
        if not os.path.exists(topicsfn):
            raise IOError(f"could not find topics file: {topicsfn}")

        # for covid:
        field2querytype = {"query": "title", "question": "description", "narrative": "narrative"}
        for k, v in field2querytype.items():
            topicfield = topicfield.replace(k, v)

        donefn = os.path.join(output_base_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"skipping Anserini SearchCollection call because path already exists: {donefn}")
            return

        # create index if it does not exist. the call returns immediately if the index does exist.
        self.index.create_index()

        os.makedirs(output_base_path, exist_ok=True)
        output_path = os.path.join(output_base_path, "searcher")

        # add stemmer and stop options to match underlying index
        indexopts = "-stemmer "
        indexopts += "none" if self.index.config["stemmer"] is None else self.index.config["stemmer"]
        if self.index.config["indexstops"]:
            indexopts += " -keepstopwords"

        index_path = self.index.get_index_path()
        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = (
            f"java -classpath {anserini_fat_jar} "
            f"-Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection "
            f"-topicreader Trec -index {index_path} {indexopts} -topics {topicsfn} -output {output_path} "
            f"-topicfield {topicfield} -inmem -threads {MAX_THREADS} {anserini_param_str}"
        )
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


class PostprocessMixin:
    def _keep_topn(self, runs, topn):
        queries = sorted(list(runs.keys()), key=lambda k: int(k))
        for q in queries:
            docs = runs[q]
            if len(docs) <= topn:
                continue
            docs = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[:topn]
            runs[q] = {k: v for k, v in docs}
        return runs

    def filter(self, run_dir, docs_to_remove=None, docs_to_keep=None, topn=None):
        if (not docs_to_keep) and (not docs_to_remove):
            raise

        for fn in os.listdir(run_dir):
            if fn == "done":
                continue

            run_fn = os.path.join(run_dir, fn)
            self._filter(run_fn, docs_to_remove, docs_to_keep, topn)
        return run_dir

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

        if topn:
            runs = self._keep_topn(runs, topn)
        Searcher.write_trec_run(runs, runfile)  # overwrite runfile

    def dedup(self, run_dir, topn=None):
        for fn in os.listdir(run_dir):
            if fn == "done":
                continue
            run_fn = os.path.join(run_dir, fn)
            self._dedup(run_fn, topn)
        return run_dir

    def _dedup(self, runfile, topn):
        runs = Searcher.load_trec_run(runfile)
        new_runs = {q: {} for q in runs}

        # use the sum of each passage score as the document score, no sorting is done here
        for q, psg in runs.items():
            for pid, score in psg.items():
                docid = pid.split(".")[0]
                new_runs[q][docid] = max(new_runs[q].get(docid, -math.inf), score)
        runs = new_runs

        if topn:
            runs = self._keep_topn(runs, topn)
        Searcher.write_trec_run(runs, runfile)


@Searcher.register
class BM25(Searcher, AnseriniSearcherMixIn):
    """ BM25 with fixed k1 and b. """

    module_name = "BM25"

    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("b", 0.4, "controls document length normalization"),
        ConfigOption("k1", 0.9, "controls term saturation"),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        """
        Runs BM25 search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        bs = [config["b"]]
        k1s = [config["k1"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = config["hits"]
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path

    def query(self, query):
        self.index.create_index()
        searcher = pysearch.SimpleSearcher(self.index.get_index_path().as_posix())
        searcher.set_bm25(self.config["k1"], self.config["b"])

        hits = searcher.search(query, k=self.config["hits"])
        return OrderedDict({hit.docid: hit.score for hit in hits})


@Searcher.register
class BM25Grid(Searcher, AnseriniSearcherMixIn):
    """ BM25 with a grid search for k1 and b. Search is from 0.1 to bmax/k1max in 0.1 increments """

    module_name = "BM25Grid"
    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("k1max", 1.0, "maximum k1 value to include in grid search (starting at 0.1)"),
        ConfigOption("bmax", 1.0, "maximum b value to include in grid search (starting at 0.1)"),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        bs = np.around(np.arange(0.1, config["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(0.1, config["k1max"] + 0.1, 0.1), 1)
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = config["hits"]
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path

    def query(self, query, b, k1):
        self.index.create_index()
        searcher = pysearch.SimpleSearcher(self.index.get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


@Searcher.register
class BM25RM3(Searcher, AnseriniSearcherMixIn):

    module_name = "BM25RM3"
    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("k1", list2str([0.65, 0.70, 0.75])),
        ConfigOption("b", list2str([0.60, 0.7])),
        ConfigOption("fbTerms", list2str([65, 70, 95, 100])),
        ConfigOption("fbDocs", list2str([5, 10, 15])),
        ConfigOption("originalQueryWeight", list2str([0.5])),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        paras = {k: " ".join(config[k].split("-")) for k in ["k1", "b", "fbTerms", "fbDocs", "originalQueryWeight"]}
        hits = str(config["hits"])

        anserini_param_str = (
            "-rm3 "
            + " ".join(f"-rm3.{k} {paras[k]}" for k in ["fbTerms", "fbDocs", "originalQueryWeight"])
            + " -bm25 "
            + " ".join(f"-bm25.{k} {paras[k]}" for k in ["k1", "b"])
            + f" -hits {hits}"
        )
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path

    def query(self, query, b, k1, fbterms, fbdocs, ow):
        self.index.create_index()
        searcher = pysearch.SimpleSearcher(self.index.get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)
        searcher.set_rm3_reranker(fb_terms=fbterms, fb_docs=fbdocs, original_query_weight=ow)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


@Searcher.register
class BM25PostProcess(BM25, PostprocessMixin):
    module_name = "BM25Postprocess"

    config_spec = [
        ConfigOption("b", 0.4, "controls document length normalization"),
        ConfigOption("k1", 0.9, "controls term saturation"),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("topn", 1000),
        ConfigOption("fields", "title"),
        ConfigOption("dedep", False),
    ]

    def query_from_file(self, topicsfn, output_path, docs_to_remove=None):
        output_path = super().query_from_file(topicsfn, output_path)

        if docs_to_remove:
            output_path = self.filter(output_path, docs_to_remove=docs_to_remove, topn=self.config["topn"])
        if self.config["dedup"]:
            output_path = self.dedup(output_path, topn=self.config["topn"])

        return output_path


@Searcher.register
class StaticBM25RM3Rob04Yang19(Searcher):
    """ Tuned BM25+RM3 run used by Yang et al. in [1]. This should be used only with a benchmark using the same folds and queries.

        [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and  the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    module_name = "bm25staticrob04yang19"

    def _query_from_file(self, topicsfn, output_path, config):
        import shutil

        outfn = os.path.join(output_path, "static.run")
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(constants["PACKAGE_PATH"] / "data" / "rob04_yang19_rm3.run", outfn)

        return output_path

    def query(self, *args, **kwargs):
        raise NotImplementedError("this searcher uses a static run file, so it cannot handle new queries")


@Searcher.register
class BM25PRF(Searcher, AnseriniSearcherMixIn):
    """
    BM25 with PRF
    """

    module_name = "BM25PRF"

    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("k1", list2str([0.65, 0.70, 0.75])),
        ConfigOption("b", list2str([0.60, 0.7])),
        ConfigOption("fbTerms", list2str([65, 70, 95, 100])),
        ConfigOption("fbDocs", list2str([5, 10, 15])),
        ConfigOption("newTermWeight", list2str([0.2, 0.25])),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    @staticmethod
    def list2str(l):
        return "-".join(str(x) for x in l)

    def _query_from_file(self, topicsfn, output_path, config):
        paras = {k: " ".join(config[k].split("-")) for k in ["k1", "b", "fbTerms", "fbDocs", "newTermWeight"]}

        hits = str(config["hits"])

        anserini_param_str = (
            "-bm25prf "
            + " ".join(f"-bm25prf.{k} {paras[k]}" for k in ["fbTerms", "fbDocs", "newTermWeight", "k1", "b"])
            + " -bm25 "
            + " ".join(f"-bm25.{k} {paras[k]}" for k in ["k1", "b"])
            + f" -hits {hits}"
        )
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class AxiomaticSemanticMatching(Searcher, AnseriniSearcherMixIn):
    """
    TODO: Add more info on retrieval method
    Also, BM25 is hard-coded to be the scoring model
    """

    module_name = "axiomatic"
    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("b", 0.4, "controls document length normalization"),
        ConfigOption("k1", 0.9, "controls term saturation"),
        ConfigOption("r", 20),
        ConfigOption("n", 30),
        ConfigOption("beta", 0.4),
        ConfigOption("top", 20),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        hits = str(config["hits"])
        conditionals = ""

        anserini_param_str = "-axiom -axiom.deterministic -axiom.r {0} -axiom.n {1} -axiom.beta {2} -axiom.top {3}".format(
            config["r"], config["n"], config["beta"], config["top"]
        )
        anserini_param_str += " -bm25 -bm25.k1 {0} -bm25.b {1} -hits {2}".format(config["k1"], config["b"], config["hits"])
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class DirichletQL(Searcher, AnseriniSearcherMixIn):
    """ Dirichlet QL with a fixed mu """

    module_name = "DirichletQL"
    dependencies = [Dependency(key="index", module="index", name="anserini")]

    config_spec = [
        ConfigOption("mu", 1000, "smoothing parameter"),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        """
        Runs Dirichlet QL search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        mus = [config["mu"]]
        mustr = " ".join(str(x) for x in mus)
        hits = config["hits"]
        anserini_param_str = f"-qld -mu {mustr} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path

    def query(self, query):
        self.index.create_index()
        searcher = pysearch.SimpleSearcher(self.index.get_index_path().as_posix())
        searcher.set_lm_dirichlet_similarity(self.config["mu"])

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


@Searcher.register
class QLJM(Searcher, AnseriniSearcherMixIn):
    """
    QL with Jelinek-Mercer smoothing
    """

    module_name = "QLJM"
    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("lam", 0.1),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-qljm -qljm.lambda {0} -hits {1}".format(config["lam"], config["hits"])

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class INL2(Searcher, AnseriniSearcherMixIn):
    """
    I(n)L2 scoring model
    """

    module_name = "INL2"
    dependencies = [Dependency(key="index", module="index", name="anserini")]
    config_spec = [
        ConfigOption("c", 0.1),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-inl2 -inl2.c {0} -hits {1}".format(config["c"], config["hits"])

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class SPL(Searcher, AnseriniSearcherMixIn):
    """
    SPL scoring model
    """

    module_name = "SPL"
    dependencies = [Dependency(key="index", module="index", name="anserini")]

    config_spec = [
        ConfigOption("c", 0.1),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-spl -spl.c {0} -hits {1}".format(config["c"], config["hits"])

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class F2Exp(Searcher, AnseriniSearcherMixIn):
    """
    F2Exp scoring model
    """

    module_name = "F2Exp"
    dependencies = [Dependency(key="index", module="index", name="anserini")]

    config_spec = [
        ConfigOption("s", 0.5),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-f2exp -f2exp.s {0} -hits {1}".format(config["s"], config["hits"])

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class F2Log(Searcher, AnseriniSearcherMixIn):
    """
    F2Log scoring model
    """

    module_name = "F2Log"
    dependencies = [Dependency(key="index", module="index", name="anserini")]

    config_spec = [
        ConfigOption("s", 0.5),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-f2log -f2log.s {0} -hits {1}".format(config["s"], config["hits"])

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path


@Searcher.register
class SDM(Searcher, AnseriniSearcherMixIn):
    """
    Sequential Dependency Model
    The scoring model is hardcoded to be BM25 (TODO: Make it configurable?)
    """

    module_name = "SDM"
    dependencies = [Dependency(key="index", module="index", name="anserini")]

    config_spec = [
        ConfigOption("b", 0.4, "controls document length normalization"),
        ConfigOption("k1", 0.9, "controls term saturation"),
        ConfigOption("tw", 0.85, "term weight"),
        ConfigOption("ow", 0.15, "ordered window weight"),
        ConfigOption("uw", 0.05, "unordered window weight"),
        ConfigOption("hits", 1000, "number of results to return"),
        ConfigOption("fields", "title"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        anserini_param_str = "-sdm -sdm.tw {0} -sdm.ow {1} -sdm.uw {2} -hits {3}".format(
            config["tw"], config["ow"], config["uw"], config["hits"]
        )
        anserini_param_str += " -bm25 -bm25.k1 {0} -bm25.b {1}".format(config["k1"], config["b"])
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path, config["fields"])

        return output_path
