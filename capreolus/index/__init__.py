import json
import logging
import math
import os
import subprocess

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.collection import Collection
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
            fields = line.strip().split()

            # is this a log line?
            # at least 5 fields should exist
            # (0) date field should be 10 digits and begin with 20. e.g. 2020-02-14
            # (3) function field should begin with [
            if len(fields) > 5 and len(fields[0]) == 10 and fields[3][0] == "[":
                # skip debug messages
                if fields[2] == "DEBUG":
                    continue

                loglevel = logging._nameToLevel.get(fields[2], 40)
                msg = " ".join(fields[3:])
            else:
                loglevel = logging._nameToLevel["WARNING"]
                msg = line.strip()

            logger.log(loglevel, "[AnseriniProcess] %s", msg)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

    def get_docs(self, doc_ids):
        if self.collection.is_large_collection:
            return self.get_documents_from_disk(doc_ids)
        else:
            return [self.getdoc(doc_id) for doc_id in doc_ids]

    # TODO: Uncomment this
    # def get_documents_from_disk(self, doc_ids):
    #     """
    #     Does not make use of the index. We use pyserini's disk traversal methods to retrieve documents. This allows
    #     us to get away with much smaller index sizes on disk, since indexes now does not have to store the document
    #     """
    #     start = time.time()
    #     logger.info("Starting to get documents from disk")
    #     document_type = self.collection.config["documents"]["type"]
    #     if document_type == "trec":
    #         ctype = "TrecCollection"
    #     elif document_type == "trecweb":
    #         ctype = "TrecwebCollection"
    #     else:
    #         # For clueweb12, document_type in yaml is the same as anserini - ClueWeb12Collection
    #         ctype = document_type
    #
    #     rootdir = self.collection.config["documents"]["path"]
    #     p = subprocess.run(
    #         ["python", get_crawl_collection_script(), rootdir, ctype],
    #         stdout=subprocess.PIPE,
    #         input=",".join(doc_ids),
    #         check=True,
    #         encoding="utf-8",
    #     )
    #     with open("{0}/disk_crawl_temp_dump.json".format(os.getenv("CAPREOLUS_CACHE", get_default_cache_dir())), "rt") as fp:
    #         fetched_docs = json.load(fp)
    #
    #     return [fetched_docs.get(doc_id, []) for doc_id in doc_ids]

    def getdoc(self, docid):
        try:
            if not hasattr(self, "index_utils") or self.index_utils is None:
                self.open()
            return self.index_utils.getTransformedDocument(docid)
        except Exception as e:
            raise

    def getdf(self, term):
        # returns 0 for missing terms
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def getidf(self, term):
        """ BM25's IDF with a floor of 0 """
        df = self.getdf(term)
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
