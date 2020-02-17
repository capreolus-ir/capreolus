import os, subprocess
from collections import defaultdict

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class Searcher(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "searcher"
    cfg = {}

    @staticmethod
    def load_trec_run(fn):
        run = defaultdict(dict)
        with open(fn, "rt", encoding="utf-8") as f:
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

    def _get_run_dir(self):
        return self.get_cache_path() / "searcher"

    def get_run_path(self):
        return self._get_run_dir() / "static.run"

    def exists(self):
        donefn = self._get_run_dir() / "done"
        return donefn.exists()

    def search(self):
        if self.exists():
            return

        self["index"].create_index()
        self._search()

        donefn = self._get_run_dir() / "done"
        with open(donefn, "wt", encoding="utf-8") as f:
            print("done", file=f)

    def _search(self):
        raise NotImplementedError()


class SDM(Searcher):
    """ a module impl """

    # idea is that we need a 2nd index (extidx) for something like calculating IDF on a larger corpus,
    # so extidx.collection should be different than the top level collection
    # ... but maybe we should not solve this in the initial attempt?
    # ... and when we do solve, something like this? Dependency(name="extidx", module="index", cls="anserini", bound="..collection..")
    # dependencies = {"index": ("index", "anserini"), "extidx": ("index", "anserini"), "extidx.collection": ("collection", "collection")}

    name = "SDM"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        ow = 0.5
        uw = 0.2


class BM25(Searcher):
    name = "BM25"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9

    def _search(self):
        # from dependencies
        index, collection = self["index"], self["index"]["collection"]
        index_path = index.get_index_path()
        topics_path, topics_type = collection.get_topics_path_and_type()
        stemmer = index.cfg["stemmer"]
        stops = "-keepStopwords" if index.cfg["indexstops"] else ""

        outpath = self.get_run_path()
        if topics_type == "trec":
            topic_reader = "Trec"
        # elif topic_type == "ClueWeb12Collection"
        #     topic_reader = "Webxml"

        bs = [self.cfg["b"]]
        k1s = [self.cfg["k1"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)

        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader {topic_reader} -index {index_path} {stops} -stemmer {stemmer} -topics {topics_path} -output {outpath} -inmem -threads {MAX_THREADS} -bm25 -b {bstr} -k1 {k1str}"

        logger.info("running bm25 search, output to %s", outpath)
        logger.debug(cmd)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        app = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        # for line in app.stdout:
        # fields = line.strip().split()
        outp = app.stdout   # check if it's verbose
        for line in outp:
            logger.info("[anserini] %s", line)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError(f"command failed: {cmd}")
