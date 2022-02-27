import os
import time
from collections import defaultdict, OrderedDict

from capreolus import ModuleBase, constants, ConfigOption
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

logger = get_logger(__name__)  # pylint: disable=invalid-name
faiss_logger = get_logger("faiss")

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

    config_spec = [ConfigOption("hits", 100, "number of results to return")]

    @staticmethod
    def load_trec_run(fn):
        # Docids in the run file appear according to decreasing score, hence it makes sense to preserve this order
        # ^ Python 3.6+ dicts preserve insertion order. Hurray!
        run = defaultdict(dict)

        with open(fn, "rt") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if len(line) > 0:
                    try:
                        qid, _, docid, rank, score, desc = line.split()
                    except ValueError as e:
                        logger.error(
                            f"Encountered malformated line when reading {fn} [Line #{i}], possibly because the writing to runfile was interruptded."
                        )
                        raise e
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn, mode="wt"):
        count = 0
        with open(outfn, mode) as outf:
            qids = sorted(preds.keys(), key=lambda k: int(k))
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    def _query_from_file(self, *args, **kwargs):
        raise NotImplementedError()

    def query_from_file(self, topicsfn, output_path, fold=None):
        output_path = self._query_from_file(topicsfn, output_path, self.config)

        return output_path

    def query(self, query, **kwargs):
        """
        search document based on given query, using parameters in config as default
        """
        config = {k: kwargs.get(k, self.config[k]) for k in self.config}

        cache_dir = self.get_cache_path()
        cache_dir.mkdir(exist_ok=True)
        topic_fn, runfile_dir = cache_dir / "topic.tsv", cache_dir / "runfiles"

        fake_qid = "1"
        with open(topic_fn, "w", encoding="utf-8") as f:
            f.write(f"{fake_qid}\t{query}")

        self._query_from_file(topic_fn, runfile_dir, config)

        runfile_fns = [f for f in os.listdir(runfile_dir) if f != "done"]
        config2runs = {}
        for runfile in runfile_fns:
            runfile_fn = runfile_dir / runfile
            runs = self.load_trec_run(runfile_fn)
            config2runs[runfile.replace("searcher_", "")] = OrderedDict(runs[fake_qid])
            os.remove(runfile_fn)  # remove it in case the file accumulate
        os.remove(runfile_dir / "done")

        return config2runs["searcher"] if len(config2runs) == 1 else config2runs


from profane import import_all_modules

from .anserini import BM25, BM25RM3, SDM

import_all_modules(__file__, __package__)
