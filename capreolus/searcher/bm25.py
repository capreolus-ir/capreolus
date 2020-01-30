import os
import shutil
import subprocess

import numpy as np

from capreolus.searcher import Searcher
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Searcher.register
class StaticBM25RM3Rob04Yang19(Searcher):
    """ Tuned BM25+RM3 run used by Yang et al. in [1]. This should be used only with a benchmark using the same folds, such as robust04.title.wsdm20demo

        [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and  the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    name = "bm25staticrob04yang19"

    def _query_index(self):
        outfn = os.path.join(self.run_path, "static.run")
        os.makedirs(self.run_path, exist_ok=True)
        shutil.copy2(os.path.join(self.collection.basepath, "rob04_yang19_rm3.run"), outfn)


@Searcher.register
class BM25Grid(Searcher):
    """ BM25 with a grid search for k1 and b. Search is from 0.1 to bmax/k1max in 0.1 increments """

    name = "bm25grid"

    @staticmethod
    def config():
        bmax = 1.0
        k1max = 1.0
        return locals().copy()  # ignored by sacred

    def _query_index(self):
        index = self.index.index_path
        outdir = self.run_path
        topics = self.collection.config["topics"]["path"]
        document_type = self.collection.config["topics"]["type"]
        if document_type == "trec":
            topic_reader = "Trec"
        elif document_type == "ClueWeb12Collection":
            topic_reader = "Webxml"

        bs = np.around(np.arange(0.1, self.pipeline_config["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(0.1, self.pipeline_config["k1max"] + 0.1, 0.1), 1)
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self.pipeline_config['stemmer']}"
        if self.pipeline_config["indexstops"]:
            indexopts += " -keepstopwords"

        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader {topic_reader} -index {index} {indexopts} -topics {topics} -output {outdir}/searcher -inmem -threads {self.pipeline_config['maxthreads']} -bm25 -b {bstr} -k1 {k1str}"
        logger.info("writing runs to %s", outdir)
        logger.debug(cmd)
        os.makedirs(outdir, exist_ok=True)
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            raise RuntimeError("command failed")


@Searcher.register
class BM25(Searcher):
    """ BM25 with fixed k1 and b. """

    name = "bm25"

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9
        return locals().copy()  # ignored by sacred

    def _query_index(self):
        index = self.index.index_path
        outdir = self.run_path
        topics = self.collection.config["topics"]["path"]
        document_type = self.collection.config["topics"]["type"]
        if document_type == "trec":
            topic_reader = "Trec"
        elif document_type == "ClueWeb12Collection":
            topic_reader = "Webxml"

        bs = [self.pipeline_config["b"]]
        k1s = [self.pipeline_config["k1"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self.pipeline_config['stemmer']}"
        if self.pipeline_config["indexstops"]:
            indexopts += " -keepstopwords"

        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader {topic_reader} -index {index} {indexopts} -topics {topics} -output {outdir}/searcher -inmem -threads {self.pipeline_config['maxthreads']} -bm25 -b {bstr} -k1 {k1str}"
        logger.info("writing runs to %s", outdir)
        logger.debug(cmd)
        os.makedirs(outdir, exist_ok=True)
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            raise RuntimeError("command failed")
