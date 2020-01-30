from capreolus.extractor import Extractor
from capreolus.tokenizer import Tokenizer
from capreolus.utils.common import padlist
from capreolus.utils.loginit import get_logger
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict

logger = get_logger(__name__)  # pylint: disable=invalid-name


class BagOfWords(Extractor):
    """ Bag of Words (or bag of trigrams when `datamode=trigram`) extractor. Used with the DSSM reranker. """

    pad = 0
    pad_tok = "<pad>"
    tokenizer_name = "anserini"

    def __init__(self, *args, **kwargs):
        super(BagOfWords, self).__init__(*args, **kwargs)
        self.itos = {self.pad: self.pad_tok}
        self.stoi = {self.pad_tok: self.pad}
        self.embeddings = self.stoi
        self.idf = defaultdict(lambda: 0)

    @staticmethod
    def config():
        datamode = "unigram"  # DSSM requires trigram vector as input
        keepstops = False
        return locals().copy()  # ignored by sacred

    def build_unigram_stoi(self, toks_list, keepstops, calculate_idf):
        nexti = len(self.stoi)
        for toks in tqdm(toks_list, desc="text to embed", unit_scale=True):
            for tok in toks:
                if tok not in self.stoi:
                    self.stoi[tok] = nexti
                    nexti += 1

                    if calculate_idf:
                        # TODO this will break if self.tokenizer is more restrictive (eg eliminates stops but index tokenizer does not)
                        # index_toks = self.index.tokenizer(self.index).tokenize(tok)
                        # index_tok = "" if len(index_toks) == 0 else index_toks[0]
                        # self.idf[tok] = self.index.getidf(index_tok)

                        # TODO: This is a temp hack. Refactor so that we query from an index where stop words are kept
                        self.idf[tok] = self.index.getidf(tok)

    def get_trigrams_for_toks(self, toks_list):
        return [("#%s#" % tok)[i : i + 3] for tok in toks_list for i in range(len(tok))]

    def build_trigram_stoi(self, toks_list, keepstops, calculate_idf):
        nexti = len(self.stoi)
        trigrams_list = []
        for toks in toks_list:
            trigrams_list.extend(self.get_trigrams_for_toks(toks))

        for trigram in trigrams_list:
            if trigram not in self.stoi:
                self.stoi[trigram] = nexti
                nexti += 1

    def build_stoi(self, toks_list, keepstops, calculate_idf):
        """
        Builds an stoi dict that stores a unique number for each token
        toks_list - An array of arrays. Each element (i.e array) in toks_list represent tokens of an entire document
        """
        if self.index is None:
            raise ValueError("Index cannot be None")

        if self.pipeline_config["datamode"] == "unigram":
            self.build_unigram_stoi(toks_list, keepstops, calculate_idf)
        elif self.pipeline_config["datamode"] == "trigram":
            self.build_trigram_stoi(toks_list, keepstops, calculate_idf)
        else:
            raise Exception("Unknown datamode")

    def build_from_benchmark(self, keepstops, *args, **kwargs):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index.")

        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(self.index, keepstops=keepstops)
        tokenizer.create()
        self.index.open()

        qids = set(self.benchmark.pred_pairs.keys()).union(self.benchmark.train_pairs.keys())
        doc_ids = set()
        for qdocs in self.benchmark.pred_pairs.values():
            doc_ids.update(qdocs)
        for qdocs in self.benchmark.train_pairs.values():
            doc_ids.update(qdocs)

        queries = self.collection.topics[self.benchmark.query_type]
        query_text_list = [queries[qid] for qid in qids]
        query_tocs_list = [tokenizer.tokenize(query) for query in query_text_list]

        logger.info("loading %s queries", len(queries))
        self.build_stoi(query_tocs_list, keepstops, True)

        self.doc_id_to_doc_toks = tokenizer.tokenizedocs(doc_ids)
        self.build_stoi([toks for doc_id, toks in self.doc_id_to_doc_toks.items()], keepstops, False)
        self.embeddings = self.stoi
        self.itos = {i: s for s, i in self.stoi.items()}
        logger.debug("The vocabulary size: %s", len(self.stoi))

    def transform_qid_posdocid_negdocid(self, q_id, posdoc_id, negdoc_id=None):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index")

        tokenizer = Tokenizer.ALL[self.tokenizer_name].get_tokenizer_instance(
            self.index, keepstops=self.pipeline_config["keepstops"]
        )
        queries = self.collection.topics[self.benchmark.query_type]

        posdoc_toks = self.doc_id_to_doc_toks.get(posdoc_id)
        query_toks = tokenizer.tokenize(queries[q_id])

        if not posdoc_toks:
            logger.debug("missing docid %s", posdoc_id)
            return None
        transformed_query = self.transform_txt(query_toks, self.pipeline_config["maxqlen"])
        idfs = [self.idf[self.itos[tok]] for tok, count in enumerate(transformed_query)]
        transformed = {
            "qid": q_id,
            "posdocid": posdoc_id,
            "negdocid": negdoc_id,
            "query": transformed_query,
            "posdoc": self.transform_txt(posdoc_toks, self.pipeline_config["maxdoclen"]),
            "query_idf": np.array(idfs, dtype=np.float32),
        }

        if negdoc_id is not None:
            negdoc_toks = self.doc_id_to_doc_toks.get(negdoc_id)
            if not negdoc_toks:
                logger.debug("missing docid %s", negdoc_id)
                return None

            transformed["negdoc"] = self.transform_txt(negdoc_toks, self.pipeline_config["maxdoclen"])

        return transformed

    def transform_txt(self, term_list, maxlen):
        nvocab = len(self.stoi)
        bog_txt = np.zeros(nvocab, dtype=np.float32)

        if self.pipeline_config["datamode"] == "unigram":
            toks = [self.stoi.get(term, 0) for term in term_list]
            tok_counts = Counter(toks)
        elif self.pipeline_config["datamode"] == "trigram":
            trigrams = self.get_trigrams_for_toks(term_list)
            toks = [self.stoi.get(trigram, 0) for trigram in trigrams]
            tok_counts = Counter(toks)
        else:
            raise Exception("Unknown datamode")

        for tok, count in tok_counts.items():
            bog_txt[tok] = count

        return bog_txt
