import json
import os
from os.path import join, exists

from capreolus.registry import Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

import numpy as np

logger = get_logger(__name__)  # pylint: disable=invalid-name


class LMDirichletWordEmbeddingsReranker(Reranker):
    """ Using language models to rank the documents with KL-div, incorporating word2vec using translation model. Tested only with PES20 benchmark.
        Dirichlet smoothing prior is set as a multiply of average documents length. The multiply can be given as input. #todo should we get mu as input as well?
    """

    name = "LMDirichletWordEmbeddings"
    dependencies = {
        "extractor": Dependency(module="extractor", name="docstatsembedding"),
        "trainer": Dependency(module="trainer", name="unsupervised"),
    }

    @staticmethod
    def config():
        similarity_threshold = 0.5
        multiplymu = 1

    def build(self):
        return self

    def get_docscore_cache_path(self):
        return self.get_cache_path() / 'docscores'

    def test(self, d):
        if not hasattr(self["extractor"], "doc_len"):
            raise RuntimeError("reranker's extractor has not been created yet. try running the task's train() method first.")
        os.makedirs(self.get_docscore_cache_path(), exist_ok=True)
        queryvocab = self["extractor"].qid_termprob[d["qid"]].keys()
        mu = self["extractor"].query_avg_doc_len[d["qid"]] * self.cfg['multiplymu']
        a = [self.score_document(queryvocab, docid, d["qid"], mu) for docid in [d["posdocid"]]]
        return a

    def score_document(self, queryvocab, docid, qid, mu):
        uid = qid.split("_")[1]
        term_scores = {}
        scoresum = 0
        for term in queryvocab:
            termscore = -1 * self.score_document_term(term, docid, qid, mu)
            if termscore != 0 and self["extractor"].domain_vocab_specific is not None:
                if term in self["extractor"].domain_term_weight: #since we might have it from the smoothing only
                    termscore *= self["extractor"].domain_term_weight[term]
            if termscore != 0 and self["extractor"].filter_query is not None:
                if self["extractor"].profile_term_weight_by == 'topic':
                    termscore *= self["extractor"].profile_term_weight[term]
                elif self["extractor"].profile_term_weight_by == 'user':
                    termscore *= self["extractor"].profile_term_weight[uid][term]

            term_scores[term] = termscore
            scoresum += termscore

        outf = join(self.get_docscore_cache_path(), f"{qid}_{docid}")
        if not exists(outf):
            with open(outf, 'w') as f:
                term_scores["OVERALL_SCORE"] = scoresum
                sorted_scores = {k: v for k, v in sorted(term_scores.items(), key=lambda item: item[1], reverse=True)}
                final_scores = {}
                for k, v in sorted_scores.items():
                    domain_term_weight = "-"
                    if self["extractor"].domain_vocab_specific is not None:
                        domain_term_weight = self["extractor"].domain_term_weight[k] if k in self[
                            "extractor"].domain_term_weight else "-"
                    prof_term_weight = "-"
                    if self["extractor"].filter_query is not None:
                        if self["extractor"].profile_term_weight_by == 'topic':
                            prof_term_weight = self["extractor"].profile_term_weight[k] if k in self[
                                "extractor"].profile_term_weight else "-"
                        elif self["extractor"].profile_term_weight_by == 'user':
                            prof_term_weight = self["extractor"].profile_term_weight[uid][k] if k in self[
                                "extractor"].profile_term_weight[uid] else "-"
                    final_scores[k] = (v, domain_term_weight, prof_term_weight)
                f.write(json.dumps(sorted_scores, indent=4))

        return scoresum
        # return -1 * sum(
        #     self.score_document_term(term, docid, qid, mu) for term in queryvocab
        # )

    def score_document_term(self, term, docid, qid, mu):
        querytp = self["extractor"].qid_termprob[qid][term]

        # here we implement a specific type of smoothing where it uses collectionP if doctf is zero TODO due to some reasons that I will have to think remember more about and also find the source and document it
        doctf = 0
        for u in self["extractor"].doc_tf[docid].keys():
            prob_term_given_u = self["extractor"].get_term_occurrence_probability(term, u, docid, self.cfg["similarity_threshold"]) 
            doctf += prob_term_given_u * self["extractor"].doc_tf[docid].get(u)

        collectiontp = self['extractor'].background_termprob(term)
        if doctf == 0:
            smoothedprob = collectiontp
        else:
            smoothedprob = (doctf + mu * collectiontp) / (self['extractor'].doc_len[docid] + mu)

        if smoothedprob == 0:
            # print("smoothed prob zero: {}".format(term)) #TODO check this further why there exists such terms that do not occure in the collection at all
            return 0

        # return querytp * (np.log10(querytp) - np.log10(smoothedprob))
        return querytp * -1 * np.log10(smoothedprob) #TODO or the complete formula? they both result in the same ranking
