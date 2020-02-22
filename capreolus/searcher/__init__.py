import os
import subprocess
from collections import defaultdict

import numpy as np

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS, PACKAGE_PATH
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

        if not self["index"].exists():
            raise Exception("Index not build")

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
        anserini_param_str = f"-bm25 -b {bstr} -k1 {k1str} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path


class BM25Grid(Searcher, AnseriniSearcherMixIn):
    """ BM25 with a grid search for k1 and b. Search is from 0.1 to bmax/k1max in 0.1 increments """

    name = "BM25Grid"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        bmax = 1.0  # maximum b value to include in grid search (starting at 0.1)
        k1max = 1.0  # maximum k1 value to include in grid search (starting at 0.1)
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
