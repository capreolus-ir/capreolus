import hashlib

import numpy as np
import torch.utils.data

from capreolus import ModuleBase, Dependency, ConfigOption, constants
from capreolus.utils.exceptions import MissingDocError
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class Sampler(ModuleBase):
    module_type = "sampler"
    requires_random_seed = True

    def prepare(self, qid_to_docids, qrels, extractor, relevance_level=1, **kwargs):
        """
        params:
        qid_to_docids: A dict of the form {qid: [list of docids to rank]}
        qrels: A dict of the form {qid: {docid: label}}
        extractor: An Extractor instance (eg: EmbedText)
        relevance_level: Threshold score below which documents are considered to be non-relevant.
        """
        self.extractor = extractor

        self.qid_to_docids = qid_to_docids
        n_unfound_queries = len([qid for qid in qid_to_docids if qid not in qrels])
        if n_unfound_queries > 0:
            logger.warning(f"There are {n_unfound_queries} missing from the qrels in total.")

        self.qid_to_reldocs = {
            qid: [docid for docid in docids if qrels.get(qid, {}).get(docid, 0) >= relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }
        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = {
            qid: [docid for docid in docids if qrels.get(qid, {}).get(docid, 0) < relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }

        self.total_samples = 0
        self.clean()

    def get_hash(self):
        raise NotImplementedError

    def get_total_samples(self):
        return self.total_samples

    def generate_samples(self):
        raise NotImplementedError


class TrainingSamplerMixin:
    def clean(self):
        # remove any ids that do not have any relevant docs or any non-relevant docs for training
        total_samples = 0  # keep tracks of the total possible number of unique training triples for this dataset
        for qid in list(self.qid_to_docids.keys()):
            posdocs = len(self.qid_to_reldocs[qid])
            negdocs = len(self.qid_to_negdocs[qid])
            if posdocs == 0 or negdocs == 0:
                logger.warning("removing training qid=%s with %s positive docs and %s negative docs", qid, posdocs, negdocs)
                del self.qid_to_reldocs[qid]
                del self.qid_to_docids[qid]
                del self.qid_to_negdocs[qid]
            else:
                total_samples += posdocs * negdocs

        self.total_samples = total_samples

    def __iter__(self):
        # when running in a worker, the sampler will be recreated several times with same seed,
        # so we combine worker_info's seed (which varies across obj creations) with our original seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # avoid reseeding the same way multiple times in case DataLoader's behavior changes
            if not hasattr(self, "_last_worker_seed"):
                self._last_worker_seed = None

            if self._last_worker_seed is None or self._last_worker_seed != worker_info.seed:
                seeds = [self.config["seed"], worker_info.seed]
                self.rng = np.random.Generator(np.random.PCG64(seeds))
                self._last_worker_seed = worker_info.seed

        return iter(self.generate_samples())


@Sampler.register
class TrainTripletSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    Samples training data triplets. Each samples is of the form (query, relevant doc, non-relevant doc)
    """

    module_name = "triplet"

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "triplet_{0}".format(key)

    def generate_samples(self):
        """
        Generates triplets infinitely.
        """
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid qids")

        while True:
            self.rng.shuffle(all_qids)

            for qid in all_qids:
                posdocid = self.rng.choice(self.qid_to_reldocs[qid])
                negdocid = self.rng.choice(self.qid_to_negdocs[qid])

                try:
                    # Convention for label - [1, 0] indicates that doc belongs to class 1 (i.e relevant
                    # ^ This is used with categorical cross entropy loss
                    yield self.extractor.id2vec(qid, posdocid, negdocid, label=[1, 0])
                except MissingDocError:
                    # at training time we warn but ignore on missing docs
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )


@Sampler.register
class TrainPairSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    Samples training data pairs. Each sample is of the form (query, doc)
    The number of generate positive and negative samples are the same.
    """

    module_name = "pair"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "pair_{0}".format(key)

    def generate_samples(self):
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            self.rng.shuffle(all_qids)
            for qid in all_qids:
                # Convention for label - [1, 0] indicates that doc belongs to class 1 (i.e relevant
                # ^ This is used with categorical cross entropy loss
                for docid in self.qid_to_reldocs[qid]:
                    yield self.extractor.id2vec(qid, docid, negid=None, label=[0, 1])
                for docid in self.qid_to_negdocs[qid]:
                    yield self.extractor.id2vec(qid, docid, negid=None, label=[1, 0])
                # REF-TODO returning all docs in a row does not make sense w/ pytorch
                #          (with TF the dataset itself is shuffled, so this is okay)
                # REF-TODO make sure always negid empty is ok


@Sampler.register
class PredSampler(Sampler, torch.utils.data.IterableDataset):
    """
    Creates a Dataset for evaluation (test) data to be used with a pytorch DataLoader
    """

    module_name = "pred"
    requires_random_seed = False

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        return "dev_{0}".format(key)

    def generate_samples(self):
        for qid, docids in self.qid_to_docids.items():
            for docid in docids:
                try:
                    if docid in self.qid_to_reldocs[qid]:
                        yield self.extractor.id2vec(qid, docid, label=[0, 1])
                    else:
                        yield self.extractor.id2vec(qid, docid, label=[1, 0])
                except MissingDocError:
                    # when predictiong we raise an exception on missing docs, as this may invalidate results
                    logger.error("got none features for prediction: qid=%s posid=%s", qid, docid)
                    raise

    def clean(self):
        total_samples = 0  # keep tracks of the total possible number of unique training triples for this dataset
        for qid in list(self.qid_to_docids.keys()):
            posdocs = len(self.qid_to_reldocs[qid])
            negdocs = len(self.qid_to_negdocs[qid])
            total_samples += posdocs * negdocs

        self.total_samples = total_samples

    def __hash__(self):
        return self.get_hash()

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return iter(self.generate_samples())

    def __len__(self):
        return sum(len(docids) for docids in self.qid_to_docids.values())

    def get_qid_docid_pairs(self):
        """
        Returns a generator for the (qid, docid) pairs. Useful if you want to sequentially access the pred pairs without
        extracting the actual content
        """
        for qid in self.qid_to_docids:
            for docid in self.qid_to_docids[qid]:
                yield qid, docid


from profane import import_all_modules

import_all_modules(__file__, __package__)
