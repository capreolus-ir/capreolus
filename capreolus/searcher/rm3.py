import os
import shutil
import subprocess

import numpy as np

from capreolus.searcher import Searcher
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Searcher.register
class BM25RM3(Searcher):
    """ BM25+RM3 with fixed parameters for b, k1 feedback terms, feedback docs, and the original query weight (b, k1, ft, fd, and ow, respectively). """

    name = "bm25rm3"

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9
        ft = 10
        fd = 10
        ow = 0.5
        return locals().copy()  # ignored by sacred

    def _query_index(self):
        index = self.index.index_path
        outdir = self.run_path
        topics = self.collection.config["topics"]["path"]
        assert self.collection.config["topics"]["type"] == "trec"

        bs = [self.pipeline_config["b"]]
        k1s = [self.pipeline_config["k1"]]
        ows = [self.pipeline_config["ow"]]
        fts = [self.pipeline_config["ft"]]
        fds = [self.pipeline_config["fd"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        owstr = " ".join(str(x) for x in ows)
        ftstr = " ".join(str(x) for x in fts)
        fdstr = " ".join(str(x) for x in fds)

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self.pipeline_config['stemmer']}"
        if self.pipeline_config["indexstops"]:
            indexopts += " -keepstopwords"

        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader Trec -index {index} {indexopts} -topics {topics} -output {outdir}/run -inmem -threads {self.pipeline_config['maxthreads']} -bm25 -b {bstr} -k1 {k1str} -rm3 -rm3.originalQueryWeight {owstr} -rm3.fbTerms {ftstr} -rm3.fbDocs {fdstr}"
        logger.info("writing runs to %s", outdir)
        logger.debug(cmd)
        os.makedirs(outdir, exist_ok=True)
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            raise RuntimeError("command failed")


@Searcher.register
class BM25RM3Grid(Searcher):
    """ BM25+RM3 with a grid search for k1, b, feedback terms, feedback docs, and the original query weight.
        - k1 and b are searched in increments of 0.1 up to k1max and bmax, respectively
        - the original query weight is searched in increments of 0.1 from 0.0 to 0.9
        - the number of feedback terms and docs is searched from 1 to ftmax/fdmax in increments of ftstep/fdstep

        Note that this grid search can take a very long time when done correctly.
        The default search space was chosen so that it will complete relatively quickly. It is NOT exhaustive.
    """

    name = "bm25rm3grid"

    @staticmethod
    def config():
        # pylint: disable=possibly-unused-variable
        bmax = 1.0
        k1max = 1.0
        ftmax = 15
        ftstep = 5
        fdmax = 15
        fdstep = 5
        return locals().copy()  # ignored by sacred

    def _query_index(self):
        index = self.index.index_path
        outdir = self.run_path
        topics = self.collection.config["topics"]["path"]
        assert self.collection.config["topics"]["type"] == "trec"

        bs = np.around(np.arange(0.1, self.pipeline_config["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(0.1, self.pipeline_config["k1max"] + 0.1, 0.1), 1)
        ows = np.around(np.arange(0.0, 1.0, 0.1), 1)
        fts = np.arange(1, self.pipeline_config["ftmax"] + self.pipeline_config["ftstep"], self.pipeline_config["ftstep"])
        fds = np.arange(1, self.pipeline_config["fdmax"] + self.pipeline_config["fdstep"], self.pipeline_config["fdstep"])

        grid_size = len(bs) * len(k1s) * len(ows) * len(fts) * len(fds)
        logger.warning("performing grid search over %s parameter combinations", grid_size)

        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        owstr = " ".join(str(x) for x in ows)
        ftstr = " ".join(str(x) for x in fts)
        fdstr = " ".join(str(x) for x in fds)

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self.pipeline_config['stemmer']}"
        if self.pipeline_config["indexstops"]:
            indexopts += " -keepstopwords"

        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader Trec -index {index} {indexopts} -topics {topics} -output {outdir}/run -inmem -threads {self.pipeline_config['maxthreads']} -bm25 -b {bstr} -k1 {k1str} -rm3 -rm3.originalQueryWeight {owstr} -rm3.fbTerms {ftstr} -rm3.fbDocs {fdstr}"
        logger.info("writing runs to %s", outdir)
        logger.debug(cmd)
        os.makedirs(outdir, exist_ok=True)
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            raise RuntimeError("command failed")


@Searcher.register
class BM25RM3Yang19(Searcher):
    """ BM25+RM3 with parameters from [1]. This should be used only with a benchmark using the same folds, such as robust04.title.wsdm20demo.
    We assume the best parameters are those described by Yang et al. and do a small search to match them with folds.

    [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    name = "bm25rm3yang19"

    @staticmethod
    def config():
        # pylint: disable=possibly-unused-variable
        return locals().copy()  # ignored by sacred

    def _query_index(self):
        index = self.index.index_path
        outdir = self.run_path
        topics = self.collection.config["topics"]["path"]
        assert self.collection.config["topics"]["type"] == "trec"

        # from https://github.com/castorini/anserini/blob/master/src/main/python/rerank/scripts/export_robust04_dataset.py#L28
        best_rm3_parameters = set([(47, 9, 0.3), (47, 9, 0.3), (47, 9, 0.3), (47, 9, 0.3), (26, 8, 0.3)])
        k1 = 0.9
        b = 0.4

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self.pipeline_config['stemmer']}"
        if self.pipeline_config["indexstops"]:
            indexopts += " -keepstopwords"

        anserini_fat_jar = Anserini.get_fat_jar()
        for fbterms, fbdocs, origw in best_rm3_parameters:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader Trec -index {index} {indexopts} -topics {topics} -output {outdir}/run_{fbterms}_{fbdocs}_{origw} -inmem -threads {self.pipeline_config['maxthreads']} -bm25 -b {b} -k1 {k1} -rm3 -rm3.fbTerms {fbterms} -rm3.fbDocs {fbdocs} -rm3.originalQueryWeight {origw}"
            logger.info("writing searcher to %s", outdir)
            logger.debug(cmd)
            os.makedirs(outdir, exist_ok=True)
            retcode = subprocess.call(cmd, shell=True)
            if retcode != 0:
                raise RuntimeError("command failed")
