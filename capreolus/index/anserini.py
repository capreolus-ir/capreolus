import json
import math
import os
import subprocess
import time
# from jnius import autoclass

from capreolus.index import Index
from capreolus.tokenizer import Tokenizer
from capreolus.utils.common import Anserini, get_default_cache_dir, get_crawl_collection_script
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Index.register
class AnseriniIndex(Index):
    """ Provides and Index module using Anserini (via the jar, not Pyserini currently). """

    name = "anserini"

    def __init__(self, *args, **kwargs):
        self.index_utils = None
        self.reader = None
        self.numdocs = None
        self.JTerm = None
        self._tokenizer = None
        self.document_disk_cache = {}

        super(AnseriniIndex, self).__init__(*args, **kwargs)

    @staticmethod
    def config():
        stemmer = "porter"
        indexstops = False
        return locals().copy()  # ignored by sacred

    def _build_index(self, config):
        outdir = self.index_path
        stops = "-keepStopwords" if config["indexstops"] else ""
        indir = self.collection.config["documents"]["path"]

        document_type = self.collection.config["documents"]["type"]
        if document_type == "trec":
            ctype = "TrecCollection"
        elif document_type == "trecweb":
            ctype = "TrecwebCollection"
        else:
            # For clueweb12, document_type in yaml is the same as anserini - ClueWeb12Collection
            ctype = document_type

        anserini_fat_jar = Anserini.get_fat_jar()
        if self.collection.is_large_collection:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {ctype} -generator JsoupGenerator -threads {config['maxthreads']} -input {indir} -index {outdir} -stemmer {config['stemmer']} {stops}"
        else:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {ctype} -generator JsoupGenerator -threads {config['maxthreads']} -input {indir} -index {outdir} -storePositions -storeDocvectors -storeTransformedDocs -stemmer {config['stemmer']} {stops}"

        logger.info("building index %s", outdir)
        logger.debug(cmd)
        os.makedirs(os.path.basename(outdir), exist_ok=True)
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            raise RuntimeError("command failed")

    def get_docs(self, doc_ids):
        if self.collection.is_large_collection:
            return self.get_documents_from_disk(doc_ids)
        else:
            return [self.getdoc(doc_id) for doc_id in doc_ids]

    def get_documents_from_disk(self, doc_ids):
        """
        Does not make use of the index. We use pyserini's disk traversal methods to retrieve documents. This allows
        us to get away with much smaller index sizes on disk, since indexes now does not have to store the document
        """
        start = time.time()
        logger.info("Starting to get documents from disk")
        document_type = self.collection.config["documents"]["type"]
        if document_type == "trec":
            ctype = "TrecCollection"
        elif document_type == "trecweb":
            ctype = "TrecwebCollection"
        else:
            # For clueweb12, document_type in yaml is the same as anserini - ClueWeb12Collection
            ctype = document_type

        rootdir = self.collection.config["documents"]["path"]
        p = subprocess.run(
            ["python", get_crawl_collection_script(), rootdir, ctype],
            stdout=subprocess.PIPE,
            input=",".join(doc_ids),
            check=True,
            encoding="utf-8",
        )
        with open("{0}/disk_crawl_temp_dump.json".format(os.getenv("CAPREOLUS_CACHE", get_default_cache_dir())), "rt") as fp:
            fetched_docs = json.load(fp)

        return [fetched_docs.get(doc_id, []) for doc_id in doc_ids]

    def getdoc(self, docid):
        try:
            if self.index_utils is None:
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

    # TODO: Remove the dependence of tokenizer on pipe. We don't really need the pipe - just the right configs
    def tokenizer(self, pipe, use_cache=False):
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.ALL["anserini"].get_tokenizer_instance(
                pipe, stemmer=pipe.cfg["stemmer"], keepstops=pipe.cfg["indexstops"], use_cache=use_cache
            )
            self._tokenizer.create()
        return self._tokenizer

    def open(self):
        from jnius import autoclass
        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        self.index_utils = JIndexUtils(self.index_path)

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(self.index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")
