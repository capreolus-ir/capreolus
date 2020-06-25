import json
import os
from os.path import join, exists

from capreolus.registry import Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class BM25Reranker(Reranker):
    """ BM25 implemented in Python as a reranker. Tested only with PES20 benchmark.
        This mainly serves as a demonstration of how non-neural methods can be prototyped as rerankers.
    """

    name = "BM25"
    dependencies = {
        "extractor": Dependency(module="extractor", name="docstats"),
        "trainer": Dependency(module="trainer", name="unsupervised"),
    }

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9

    def build(self):
        return self
    
    def get_docscore_cache_path(self):
        return self.get_cache_path() / 'docscores'

    def test(self, d):
        if not hasattr(self["extractor"], "doc_len"):
            raise RuntimeError("reranker's extractor has not been created yet. try running the task's train() method first.")


        # query = self["extractor"].qid2toks[d["qid"]]
        # avg_doc_len = self["extractor"].query_avg_doc_len[d["qid"]]
        # return [self.score_document(query, d["qid"], docid, avg_doc_len) for docid in [d["posdocid"]]]
        querytf = self["extractor"].qid_termprob[d["qid"]]
        querylen = self["extractor"].qidlen[d["qid"]]
        avg_doc_len = self["extractor"].query_avg_doc_len[d["qid"]]
        return [self.score_document_tf(querytf, querylen, d["qid"], docid, avg_doc_len) for docid in [d["posdocid"]]]

    def score_document(self, query, qid, docid, avg_doc_len):
        # TODO is it correct to skip over terms that don't appear to be in the idf vocab?
        term_scores = {}
        accumulated_scores = {}
        scoresum = 0
        for term in query:
            temp = self.score_document_term(term, docid, avg_doc_len)
            if term not in accumulated_scores:
                accumulated_scores[term] = 0
            term_scores[term] = temp
            accumulated_scores[term] += temp
            scoresum += temp
        
        os.makedirs(self.get_docscore_cache_path(), exist_ok=True)
        outf = join(self.get_docscore_cache_path(), f"{qid}_{docid}")
        # if not exists(outf):
        with open(outf, 'w') as f:
            term_scores["OVERALL_SCORE"] = "-"
            accumulated_scores["OVERALL_SCORE"] = scoresum
            sorted_acc_scores = {k: v for k, v in sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)}
            final_scores = {k: (v, term_scores[k]) for k, v in sorted_acc_scores.items()}
            f.write(json.dumps(final_scores, indent=4))
        
        return scoresum
        #return sum(self.score_document_term(term, docid, avg_doc_len) for term in query)

    def score_document_tf(self, querytf, querylen, qid, docid, avg_doc_len):
        # TODO is it correct to skip over terms that don't appear to be in the idf vocab?
        term_scores = {}
        accumulated_scores = {}
        scoresum = 0
        for term, tf in querytf.items():
            termsccore = self.score_document_term(term, docid, avg_doc_len)
            term_scores[term] = termsccore
            accumulated_scores[term] = termsccore * round(tf * querylen)
            scoresum += accumulated_scores[term]

        os.makedirs(self.get_docscore_cache_path(), exist_ok=True)
        outf = join(self.get_docscore_cache_path(), f"{qid}_{docid}")
        # if not exists(outf):
        with open(outf, 'w') as f:
            term_scores["OVERALL_SCORE"] = "-"
            accumulated_scores["OVERALL_SCORE"] = scoresum
            sorted_acc_scores = {k: v for k, v in
                                 sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)}
            final_scores = {k: (v, term_scores[k]) for k, v in sorted_acc_scores.items()}
            f.write(json.dumps(final_scores, indent=4))

        return scoresum
        # return sum(self.score_document_term(term, docid, avg_doc_len) for term in query)

    def score_document_term(self, term, docid, avg_doc_len):
        tf = self["extractor"].doc_tf[docid].get(term, 0)
        numerator = tf * (self.cfg["k1"] + 1)
        denominator = tf + self.cfg["k1"] * (1 - self.cfg["b"] + self.cfg["b"] * (self["extractor"].doc_len[docid] / avg_doc_len))

        idf = self["extractor"].background_idf(term)

        return idf * (numerator / denominator)
