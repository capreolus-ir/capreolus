import hashlib
import os

from capreolus import ModuleBase, get_logger

logger = get_logger(__name__)


class Extractor(ModuleBase):
    """Base class for Extractor modules. The purpose of an Extractor is to convert queries and documents to a representation suitable for use with a :class:`~capreolus.reranker.Reranker` module.

    Modules should provide:
        - an ``id2vec(qid, posid, negid=None)`` method that converts the given query and document ids to an appropriate representation
    """

    module_type = "extractor"
    requires_random_seed = True

    def _extend_stoi(self, toks_list, calc_idf=False):
        if not self.stoi:
            logger.warning("extending stoi while it's not yet instantiated")
            self.stoi = {}
        # TODO is this warning working correctly?
        if calc_idf and not self.idf:
            logger.warning("extending idf while it's not yet instantiated")
            self.idf = {}
        if calc_idf and not hasattr(self, "index"):
            logger.warning("requesting calculating idf yet index is not available, set calc_idf to False")
            calc_idf = False

        n_words_before = len(self.stoi)
        for toks in toks_list:
            toks = [toks] if isinstance(toks, str) else toks
            for tok in toks:
                if tok not in self.stoi:
                    self.stoi[tok] = len(self.stoi)
                if calc_idf and tok not in self.idf:
                    self.idf[tok] = self.index.get_idf(tok)

        logger.debug(f"added {len(self.stoi)-n_words_before} terms to the stoi of extractor {self.module_name}")

    def cache_state(self, qids, docids):
        raise NotImplementedError

    def load_state(self, qids, docids):
        raise NotImplementedError

    def get_state_cache_file_path(self, qids, docids):
        """
        Returns the path to the cache file used to store the extractor state, regardless of whether it exists or not
        """
        sorted_qids = sorted(qids)
        sorted_docids = sorted(docids)
        return self.get_cache_path() / hashlib.md5(str(sorted_qids + sorted_docids).encode("utf-8")).hexdigest()

    def is_state_cached(self, qids, docids):
        """
        Returns a boolean indicating whether the state corresponding to the qids and docids passed has already
        been cached
        """
        file_path = self.get_state_cache_file_path(qids, docids)
        logger.debug("Looking for extractor cache at {}".format(file_path))
        return os.path.exists(file_path)

    def _build_vocab(self, qids, docids, topics):
        raise NotImplementedError

    def build_from_benchmark(self, *args, **kwargs):
        raise NotImplementedError

    def id2vec(self, qid, posdocid, negdocid=None, label=None):
        """
        Creates a feature from the (qid, docid) pair.
        If negdocid is supplied, that's also included in the feature (needed for training with pairwise hinge loss)
        Label is a vector of shape [num_classes], and is supplied only when using pointwise training (i.e cross entropy)
        When using pointwise samples, negdocid is None, and label is either [0, 1] or [1, 0] depending on whether the
        document represented by posdocid is relevant or irrelevant respectively.
        """
        raise NotImplementedError


from profane import import_all_modules


import_all_modules(__file__, __package__)
