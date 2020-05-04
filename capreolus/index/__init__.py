import logging
import math
import os
import subprocess
from os.path import join

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Index(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "index"
    dependencies = {"collection": Dependency(module="collection")}

    def get_index_path(self):
        return self.get_cache_path() / "index"

    def exists(self):
        donefn = self.get_index_path() / "done"
        return donefn.exists()

    def create_index(self):
        if self.exists():
            return

        self._create_index()
        donefn = self.get_index_path() / "done"
        with open(donefn, "wt") as donef:
            print("done", file=donef)

    def _create_index(self):
        raise NotImplementedError()

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def get_docs(self, doc_ids):
        raise NotImplementedError()


def get_cache_path(self):
    print(self.get_cache_path())


class AnseriniIndex(Index):
    name = "anserini"
    commands = {"cache_path": get_cache_path}

    @staticmethod
    def config():
        indexstops = False
        stemmer = "porter"

    def _create_index(self):
        outdir = self.get_index_path()
        stops = "-keepStopwords" if self.cfg["indexstops"] else ""

        collection_path, document_type, generator_type = self["collection"].get_path_and_types()

        anserini_fat_jar = Anserini.get_fat_jar()
        if self["collection"].is_large_collection:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {document_type} -generator {generator_type} -threads {MAX_THREADS} -input {collection_path} -index {outdir} -stemmer {self.cfg['stemmer']} {stops}"
        else:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {document_type} -generator {generator_type} -threads {MAX_THREADS} -input {collection_path} -index {outdir} -storePositions -storeDocvectors -storeTransformedDocs -stemmer {self.cfg['stemmer']} {stops}"

        logger.info("building index %s", outdir)
        logger.debug(cmd)
        os.makedirs(os.path.basename(outdir), exist_ok=True)

        app = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

        # Anserini output is verbose, so ignore DEBUG log lines and send other output through our logger
        for line in app.stdout:
            Anserini.filter_and_log_anserini_output(line, logger)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

    def get_docs(self, doc_ids):
        # if self.collection.is_large_collection:
        #     return self.get_documents_from_disk(doc_ids)

        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        try:
            if not hasattr(self, "index_utils") or self.index_utils is None:
                self.open()
            return self.index_utils.getTransformedDocument(docid)
        except Exception as e:
            raise

    def get_df(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def get_idf(self, term):
        """ BM25's IDF with a floor of 0 """
        df = self.get_df(term)
        idf = (self.numdocs - df + 0.5) / (df + 0.5)
        idf = math.log(1 + idf)
        return max(idf, 0)

    def open(self):
        from jnius import autoclass

        index_path = self.get_index_path().as_posix()

        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        self.index_utils = JIndexUtils(index_path)

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")

class AnseriniCorpusIndex(Index):
    name = "anserinicorpus"
    corpusdir = '/GW/NeuralIR/nobackup/' #TODO hard codeed

    @staticmethod
    def config():
        indexcorpus = 'anserini0.9-index.clueweb09.englishonly.nostem.stopwording'# todo: this was the only one which was working now

    def get_index_path(self):
        return join(self.corpusdir, self.cfg['indexcorpus'])

    def get_df(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def get_tf(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.totalTermFreq(jterm)

    # def get_idf(self, term): # for now I wrote another one which doesn't have the last line in the extractor, we could use this as well if this is better.
    #     """ BM25's IDF with a floor of 0 """
    #     df = self.get_df(term)
    #     idf = (self.numdocs - df + 0.5) / (df + 0.5)
    #     idf = math.log(1 + idf)
    #     return max(idf, 0)

    def open(self):
        from jnius import autoclass

        index_path = self.get_index_path()

        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        self.index_utils = JIndexUtils(index_path)

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")
        self.numterms = self.reader.getSumTotalTermFreq("contents");
