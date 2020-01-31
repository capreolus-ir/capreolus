import os
import time

import capnp

from capreolus.utils.common import register_component_module, import_component_modules, args_to_key
from capreolus.utils.cache_capnp import Document

from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Tokenizer:
    """ Module for tokenizing text. Convenience method tokenizedocs retrieves, tokenizes, and caches documents from an Index. """

    ALL = {}
    cached_instances = {}

    def __init__(self, index):
        """
        :param index: Instance of an Index class. This is used to fetch entire documents from disk. The index also
        determines the location of the cache
        eg: see tokenizedoc()
        """
        self.index = index
        self.dirty = False
        self.cache = {}
        self.cache_path = None

    @classmethod
    def get_tokenizer_instance(cls, index, *args, **kwargs):
        cache_key = str(index) + str(args) + str(sorted([(k, v) for k, v in kwargs.items()]))
        if cache_key in cls.cached_instances:
            return cls.cached_instances[cache_key]

        cls.cached_instances[cache_key] = cls(index, *args, **kwargs)
        return cls.cached_instances[cache_key]

    @classmethod
    def register(cls, subcls):
        return register_component_module(cls, subcls)

    def tokenizedocs(self, doc_ids):
        doc_id_to_toks = {doc_id: self.cache[doc_id] for doc_id in doc_ids if doc_id in self.cache}
        doc_ids_to_fetch = [doc_id for doc_id in doc_ids if doc_id not in self.cache]

        if doc_ids_to_fetch:
            fetched_docs = self.index.get_docs(doc_ids_to_fetch)
            if not fetched_docs:
                toks_list = []
            else:
                toks_list = [self.tokenize(doc) for doc in fetched_docs]
                self.dirty = True

            for doc_id, doc_toks in zip(doc_ids_to_fetch, toks_list):
                doc_id_to_toks[doc_id] = doc_toks
                self.cache[doc_id] = doc_toks

        return doc_id_to_toks

    def tokenizedoc(self, docid):
        if docid in self.cache:
            return self.cache[docid]
        self.dirty = True

        doc = self.index.getdoc(docid)
        if not doc:
            toks = []
        else:
            toks = self.tokenize(doc)
        self.cache[docid] = toks
        return toks

    def initialize_cache(self, params):
        key = args_to_key(self.name, params)
        self.cache_path = self.doc_cache(key)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.cache = {}
        self.load_cache()

    def doc_cache(self, key):
        # key should be relative to the index_path (e.g., tokenizer options)
        s = os.path.join(self.index.index_key, "docs", key, "document_capnp.cache")
        return s

    def write_cache(self):
        if not self.dirty:
            return

        with open(self.cache_path, "w+b") as outf:
            # docid can refer to a stringified list of doc_ids as well
            for docid, txt in self.cache.items():
                doc = Document.new_message(txt=" ".join(txt), docid=docid)
                doc.write(outf)

        self.dirty = False

    def load_cache(self):
        if not os.path.exists(self.cache_path):
            return

        load_start = time.time()
        with open(self.cache_path, "rb") as inf:
            for doc in Document.read_multiple(inf):
                self.cache[doc.docid] = doc.txt.split()
        logger.info("Loading tokenizer cache took {0} seconds".format(time.time() - load_start))


@Tokenizer.register
class AnseriniTokenizer(Tokenizer):
    """ Tokenize text using the Lucene tokenizer used by Anserini """

    name = "anserini"

    def __init__(self, index, stemmer="none", keepstops=False, use_cache=True):
        super().__init__(index)
        self.params = {"stemmer": stemmer, "keepstops": keepstops}

        if use_cache:
            self.initialize_cache(self.params)
        self.create()

    def create(self):
        from jnius import autoclass

        stemmer = self.params["stemmer"]
        keepstops = self.params["keepstops"]

        if keepstops:
            emptyjchar = autoclass("org.apache.lucene.analysis.CharArraySet").EMPTY_SET
            self.analyzer = autoclass("io.anserini.analysis.EnglishStemmingAnalyzer")(stemmer, emptyjchar)
        else:
            self.analyzer = autoclass("io.anserini.analysis.EnglishStemmingAnalyzer")(stemmer)

        self._tokenize = autoclass("io.anserini.analysis.AnalyzerUtils").tokenize

    def tokenize(self, s):
        if not s:
            return []
        return self._tokenize(self.analyzer, s).toArray()


import_component_modules("tokenizer")
