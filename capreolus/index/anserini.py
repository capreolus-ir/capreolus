import math
import os
import subprocess

from capreolus import ConfigOption, constants, get_logger
from capreolus.utils.common import Anserini

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]


@Index.register
class AnseriniIndex(Index):
    module_name = "anserini"
    config_spec = [
        ConfigOption("indexstops", False, "should stopwords be indexed? (if False, stopwords are removed)"),
        ConfigOption("stemmer", "porter", "stemmer: porter, krovetz, or none"),
    ]

    def _create_index(self):
        outdir = self.get_index_path()
        collection_path, document_type, generator_type = self.collection.get_path_and_types()
        anserini_fat_jar = Anserini.get_fat_jar()

        cmd = [
            "java",
            "-classpath",
            anserini_fat_jar,
            "-Xms512M",
            "-Xmx31G",
            "-Dapp.name='IndexCollection'",
            "io.anserini.index.IndexCollection",
            "-collection",
            document_type,
            "-generator",
            generator_type,
            "-threads",
            str(MAX_THREADS),
            "-input",
            collection_path,
            "-index",
            outdir,
            "-stemmer",
            "none" if self.config["stemmer"] is None else self.config["stemmer"],
        ]

        if self.config["indexstops"]:
            cmd += ["-keepStopwords"]

        if not self.collection.is_large_collection:
            cmd += [
                "-storePositions",
                "-storeDocvectors",
                "-storeContents",
            ]

        logger.info("building index %s", outdir)
        logger.debug(cmd)
        os.makedirs(os.path.basename(outdir), exist_ok=True)

        app = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

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
            if not hasattr(self, "index_reader_utils") or self.index_reader_utils is None:
                self.open()
            return self.index_reader_utils.documentContents(self.reader, self.JString(docid))
        except Exception as e:
            raise

    def convert_lucene_ids_to_doc_ids(self, lucene_ids):
        return [self.convert_lucene_id_to_doc_id(lucene_id) for lucene_id in lucene_ids]

    def convert_lucene_id_to_doc_id(self, lucene_id):
        if not hasattr(self, "index_utils") or self.index_utils is None or not hasattr(self, "reader") or self.reader is None:
            self.open()

        return self.index_reader_utils.convertLuceneDocidToDocid(self.reader, lucene_id)

    def convert_doc_id_to_lucene_id(self, doc_id):
        if not hasattr(self, "index_utils") or self.index_utils is None or not hasattr(self, "reader") or self.reader is None:
            self.open()

        return self.index_reader_utils.convertDocidToLuceneDocid(self.reader, doc_id)

    def get_all_docids_in_collection(self):
        from jnius import autoclass

        self.create_index()

        anserini_index_path = self.get_index_path().as_posix()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        # TODO: Add check for deleted rows in the index
        all_doc_ids = [self.convert_lucene_id_to_doc_id(i) for i in range(0, anserini_index_reader.maxDoc())]

        return all_doc_ids

    def get_anserini_index_reader(self):
        from jnius import autoclass

        self.create_index()

        anserini_index_path = self.get_index_path().as_posix()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        return anserini_index_reader

    def get_df(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def get_idf(self, term):
        """BM25's IDF with a floor of 0"""
        df = self.get_df(term)
        idf = (self.numdocs - df + 0.5) / (df + 0.5)
        idf = math.log(1 + idf)
        return max(idf, 0)

    def open(self):
        from jnius import autoclass

        index_path = self.get_index_path().as_posix()

        JIndexReaderUtils = autoclass("io.anserini.index.IndexReaderUtils")
        self.index_reader_utils = JIndexReaderUtils()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")
        self.JString = autoclass("java.lang.String")
