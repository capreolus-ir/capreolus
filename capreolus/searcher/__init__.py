import os
from collections import defaultdict, OrderedDict

from capreolus import ModuleBase, constants
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt
from capreolus.utils.common import OrderedDefaultDict

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]


def list2str(l, delimiter="-"):
    return delimiter.join(str(x) for x in l)


class Searcher(ModuleBase):
    """Base class for Searcher modules. The purpose of a Searcher is to query a collection via an :class:`~capreolus.index.Index` module.

    Similar to Rerankers, Searchers return a list of documents and their relevance scores for a given query.
    Searchers are unsupervised and efficient, whereas Rerankers are supervised and do not use an inverted index directly.

    Modules should provide:
        - a ``query(string)`` and a ``query_from_file(path)`` method that return document scores
    """

    module_type = "searcher"

    @staticmethod
    def load_trec_run(fn):
        # Docids in the run file appear according to decreasing score, hence it makes sense to preserve this order
        run = OrderedDefaultDict()

        with open(fn, "rt") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    qid, _, docid, rank, score, desc = line.split(" ")
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn):
        count = 0
        with open(outfn, "wt") as outf:
            qids = sorted(preds.keys(), key=lambda k: int(k))
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    def _query_from_file(self, topicsfn, output_path, cfg):
        raise NotImplementedError()

    def query_from_file(self, topicsfn, output_path):
        return self._query_from_file(topicsfn, output_path, self.config)

    def query(self, query, **kwargs):
        raise NotImplementedError()


from profane import import_all_modules

from .anserini import BM25, BM25RM3, SDM

import_all_modules(__file__, __package__)
