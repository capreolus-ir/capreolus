import hashlib
from collections import defaultdict

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
        qid_to_docids: A dict of the form {qid: [list of docids to rank]} or of the form {qid: {docid: score}}

        qrels: A dict of the form {qid: {docid: label}}
        extractor: An Extractor instance (eg: EmbedText)
        relevance_level: Threshold score below which documents are considered to be non-relevant.
        """
        self.extractor = extractor

        # remove qids from qid_to_docids that do not have relevance labels in the qrels
        self.qid_to_docids = {qid: docids for qid, docids in qid_to_docids.items() if qid in qrels}
        if len(self.qid_to_docids) != len(qid_to_docids):
            logger.warning(
                f"skipping qids that were missing from the qrels: {len(qid_to_docids.keys() - self.qid_to_docids.keys())} in total."
            )

        self.qid_to_reldocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) >= relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }
        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid, 0) < relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }

        count = 0
        for qid, reldocs in self.qid_to_reldocs.items():
            if len(reldocs):
                count += 1

        logger.info("{} out of {} queries had a relevant doc in the BM25 results".format(count, len(self.qid_to_reldocs)))

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
                # logger.warning("removing training qid=%s with %s positive docs and %s negative docs", qid, posdocs, negdocs)
                del self.qid_to_reldocs[qid]
                del self.qid_to_docids[qid]
                del self.qid_to_negdocs[qid]
            else:
                total_samples += posdocs * negdocs

        self.total_samples = total_samples

        logger.debug("Done cleaning the sampler")

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
class QrelTrainTripletSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    Samples positive docs from the qrels
    """

    module_name = "qreltriplet"

    def prepare(self, bm25_run, qrels, extractor, relevance_level=1, **kwargs):
        self.extractor = extractor
        self.qid_to_docids = bm25_run

        self.qid_to_reldocs = defaultdict(list)
        for qid, docid_to_label in qrels.items():
            if qid not in bm25_run:
                continue
            for docid, label in docid_to_label.items():
                if label >= relevance_level:
                    self.qid_to_reldocs[qid].append(docid)

        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = defaultdict(list)
        for qid, docid_to_score in bm25_run.items():
            if qid not in qrels:
                continue
            for docid, score in docid_to_score.items():
                # Skip the relevant docs
                if qrels[qid].get(docid, 0) < relevance_level:
                    self.qid_to_negdocs[qid].append(docid)

        self.total_samples = 0
        self.clean()

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "qreltriplet_{0}".format(key)

    def generate_samples(self):
        """
        Generates triplets infinitely.
        """
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid qids")

        while True:
            qid = self.rng.choice(all_qids)
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
    The number of generated positive and negative samples are the same.
    We alternate between posdoc and negdocs. This is required for RepBERT.
    """

    module_name = "pair"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "pair_{0}".format(key)

    def generate_samples(self):
        all_qids = sorted([qid for qid in self.qid_to_reldocs if len(self.qid_to_reldocs[qid]) > 0])

        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            qid = self.rng.choice(all_qids)
            posdocid = self.rng.choice(self.qid_to_reldocs[qid])
            negdocid = self.rng.choice(self.qid_to_negdocs[qid])
            yield self.extractor.id2vec_for_train(qid, posdocid, negid=None, label=[0, 1],
                                                  reldocs=set(self.qid_to_reldocs[qid]))

            yield self.extractor.id2vec_for_train(qid, negdocid, negid=None, label=[1, 0],
                                                  reldocs=set(self.qid_to_reldocs[qid]))


@Sampler.register
class TrainPairSamplerRobust04passages(TrainPairSampler):
    """
    Samples training data pairs. Each sample is of the form (query, doc)
    The number of generated positive and negative samples are the same.
    We alternate between posdoc and negdocs. This is required for RepBERT.
    """

    module_name = "pairrobust04passages"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "pairrobust04passages_{0}".format(key)

    def prepare(self, qid_to_docids, qrels, extractor, relevance_level=1, separator="_", **kwargs):
        """
        params:
        qid_to_docids: A dict of the form {qid: [list of docids to rank]} or of the form {qid: {docid: score}}

        qrels: A dict of the form {qid: {docid: label}}
        extractor: An Extractor instance (eg: EmbedText)
        relevance_level: Threshold score below which documents are considered to be non-relevant.
        separator: Some datasets (eg: robust04passages) has docids in a <docid>_<passageid> format while the qrels only
        use the docid part. In these cases, we have to ignore the passageid part - the separator helps with this.
        """
        self.extractor = extractor

        # remove qids from qid_to_docids that do not have relevance labels in the qrels
        self.qid_to_docids = {qid: docids for qid, docids in qid_to_docids.items() if qid in qrels}
        if len(self.qid_to_docids) != len(qid_to_docids):
            logger.warning(
                f"skipping qids that were missing from the qrels: {len(qid_to_docids.keys() - self.qid_to_docids.keys())} in total."
            )

        self.qid_to_reldocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid.split(separator)[0], 0) >= relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }
        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = {
            qid: [docid for docid in docids if qrels[qid].get(docid.split(separator)[0], 0) < relevance_level]
            for qid, docids in self.qid_to_docids.items()
        }

        count = 0
        for qid, reldocs in self.qid_to_reldocs.items():
            if len(reldocs):
                count += 1

        logger.info("{} out of {} queries had a relevant doc in the BM25 results".format(count, len(self.qid_to_reldocs)))

        self.total_samples = 0
        self.clean()


@Sampler.register
class QrelTrainPairSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    module_name = "qrelpair"
    dependencies = []

    def prepare(self, bm25_run, qrels, extractor, relevance_level=1, **kwargs):
        self.extractor = extractor
        self.qid_to_docids = bm25_run

        self.qid_to_reldocs = defaultdict(list)
        for qid, docid_to_label in qrels.items():
            if qid not in bm25_run:
                continue
            for docid, label in docid_to_label.items():
                if label >= relevance_level:
                    self.qid_to_reldocs[qid].append(docid)

        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = defaultdict(list)
        for qid, docid_to_score in bm25_run.items():
            doc_id_score_list = []
            for docid, score in docid_to_score.items():
                if qrels[qid].get(docid, 0) < relevance_level:
                    doc_id_score_list.append((docid, score))

            sorted_according_to_score = sorted(doc_id_score_list, key=lambda x: x[1], reverse=True)
            # Arbitrarily choosing the top 20 non-relevand docs.
            self.qid_to_negdocs[qid] = sorted_according_to_score[:20]

        self.total_samples = 0
        self.clean()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "qrelpair{0}".format(key)

    def generate_samples(self):
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            qid = self.rng.choice(all_qids)
            posdocid = self.rng.choice(self.qid_to_reldocs[qid])
            negdocid = self.rng.choice(self.qid_to_negdocs[qid])
            yield self.extractor.id2vec_for_train(qid, posdocid, negid=None, label=[0, 1],
                                                      reldocs=set(self.qid_to_reldocs[qid]))
            yield self.extractor.id2vec_for_train(qid, negdocid, negid=None, label=[1, 0],
                                                  reldocs=set(self.qid_to_reldocs[qid]))


@Sampler.register
class QrelTrainPairSamplerRobust04PassagesDocT5(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    WARNING: Works only with the qrels in qrels.robust04doct5title.txt
    """
    module_name = "qrelpairrobust04passagesdoct5"
    dependencies = []

    def prepare(self, bm25_run, qrels, extractor, relevance_level=1, separator="_", **kwargs):
        self.extractor = extractor
        self.qid_to_docids = bm25_run

        self.qid_to_reldocs = defaultdict(list)
        for qid, passageid_to_label in qrels.items():
            if qid not in bm25_run:
                continue
            for passageid, label in passageid_to_label.items():
                self.qid_to_reldocs[qid].append(passageid)

        # TODO option to include only negdocs in a top k
        self.qid_to_negdocs = defaultdict(list)
        for qid, passageid_to_score in bm25_run.items():
            passage_id_score_list = []
            for passage_id, score in passageid_to_score.items():
                if qrels[qid].get(passage_id, 0) < relevance_level:
                    passage_id_score_list.append((passage_id, score))

            sorted_according_to_score = sorted(passage_id_score_list, key=lambda x: x[1], reverse=True)
            # Arbitrarily choosing the top 20 non-relevand docs.
            self.qid_to_negdocs[qid] = sorted_according_to_score[:20]

        self.total_samples = 0
        self.clean()

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "qrelpairrobust04passages_{0}".format(key)

    def generate_samples(self):
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            qid = self.rng.choice(all_qids)
            posdocid = self.rng.choice(self.qid_to_reldocs[qid])
            negdocid = self.rng.choice(self.qid_to_negdocs[qid])
            # The `posdocid` is of the format docid_passageid. This might not exist in the index - hence the try block
            try:
                yield self.extractor.id2vec_for_train(qid, posdocid, negid=None, label=[0, 1],
                                                      reldocs=set(self.qid_to_reldocs[qid]))
            except MissingDocError:
                continue

            yield self.extractor.id2vec_for_train(qid, negdocid, negid=None, label=[1, 0],
                                                  reldocs=set(self.qid_to_reldocs[qid]))


@Sampler.register
class ReldocAsQuerySampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    Same as TrainPairSampler, but relevant docs too can be queries
    """
    module_name = "reldocpair"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "reldocpair_{0}".format(key)

    def generate_samples(self):
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            qid = self.rng.choice(all_qids)
            reldocs = self.qid_to_reldocs[qid]
            posdocid, another_posdocid = self.rng.choice(self.qid_to_reldocs[qid], 2, replace=False)
            negdocid = self.rng.choice(self.qid_to_negdocs[qid])

            yield self.extractor.id2vec_for_train_reldoc_as_query(qid, posdocid, another_posdocid, reldocs=reldocs)
            yield self.extractor.id2vec_for_train_reldoc_as_query(qid, posdocid, negdocid, reldocs=reldocs)


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
        for qid, docids in self.qid_to_docids.items():
            total_samples += len(docids)

        logger.debug("There are {} samples in the PredSampler".format(total_samples))
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


@Sampler.register
class CollectionSampler(Sampler, torch.utils.data.IterableDataset):
    """
    Goes throw every document in the collection. One use case - allows you to encode every document in the collection for ANN search. Does not make use of queries
    """
    module_name = "collection"
    requires_random_seed = False

    def prepare(self, docids, qrels, extractor, relevance_level=1, **kwargs):
        assert qrels is None, "Do not pass qrels to the collection sampler. Pass None"
        self.extractor = extractor
        self.docids = docids

    def get_hash(self):
        sorted_rep = sorted(self.docids)
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        return "collection_{0}".format(key)

    def generate_samples(self):
        for docid in self.docids:
            try:
                yield self.extractor.id2vec(None, docid)
            except MissingDocError:
                logger.info("Doc {} was missing".format(docid))

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return iter(self.generate_samples())

    def __len__(self):
        return len(self.docids)


@Sampler.register
class ResidualTrainPairSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    Samples training data pairs. Each sample is of the form (query, doc)
    The number of generated positive and negative samples are the same.
    We alternate between posdoc and negdocs. This is required for RepBERT.
    """

    module_name = "residualpair"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "residualpair_{0}".format(key)

    def generate_samples(self):
        lambda_train = 0.1
        epsilon = 1

        # TODO: Fix this naming.
        trec_run = self.qid_to_docids

        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")
        while True:
            qid = self.rng.choice(all_qids)
            posdocid = self.rng.choice(self.qid_to_reldocs[qid])
            negdocid = self.rng.choice(self.qid_to_negdocs[qid])
            data = self.extractor.id2vec_for_train(qid, posdocid, negid=None, label=1,
                                                  reldocs=set(self.qid_to_reldocs[qid]))
            data["residual"] = epsilon - lambda_train * (trec_run[qid][posdocid] - trec_run[qid][negdocid])
            yield data

            data = self.extractor.id2vec_for_train(qid, negdocid, negid=None, label=0,
                                                  reldocs=set(self.qid_to_reldocs[qid]))
            data["residual"] = epsilon - lambda_train * (trec_run[qid][posdocid] - trec_run[qid][negdocid])
            yield data


@Sampler.register
class ResidualTripletSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    """
    TODO: This needs a re-write to make it work with RepBERT:
    1. RepBERT models expect pairs right now. Change it to make it accept triplets
    """
    module_name = "residualtriplet"

    def prepare(self, trec_run, qrels, extractor, relevance_level=1, **kwargs):
        self.trec_run = trec_run
        self.extractor = extractor
        self.qids = sorted([qid for qid in qrels.keys()])

        qid_to_reldocs = defaultdict(list)
        # We call these "noise docs" since we have no relevance labels for them.
        qid_to_negdocs = defaultdict(list)

        for qid, doc_id_to_score in trec_run.items():
            if qid not in qrels:
                continue
            for doc_id, score in doc_id_to_score.items():
                if doc_id in qrels[qid] and qrels[qid][doc_id] >= relevance_level:
                    qid_to_reldocs[qid].append(doc_id)
                else:
                    qid_to_negdocs[qid].append(doc_id)

        self.qid_to_reldocs = qid_to_reldocs
        self.qid_to_negdocs = qid_to_negdocs

    def get_hash(self):
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(self.trec_run))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        return "residual_{0}".format(key)

    def generate_samples(self):
        lambda_train = 0.1
        epsilon = 1
        all_qids = self.qids
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid qids")

        while True:
            self.rng.shuffle(all_qids)

            for qid in all_qids:
                if qid not in self.qid_to_reldocs:
                    continue

                posdocid = self.rng.choice(self.qid_to_reldocs[qid])
                negdocid = self.rng.choice(self.qid_to_negdocs[qid])

                try:
                    # Convention for label - [1, 0] indicates that doc belongs to class 1 (i.e relevant
                    # ^ This is used with categorical cross entropy loss
                    data = self.extractor.id2vec(qid, posdocid, negdocid, label=[1, 0])

                    # This is equation 4 in the CLEAR paper
                    data["residual"] = epsilon - lambda_train * (self.trec_run[qid][posdocid] - self.trec_run[qid][negdocid])

                    yield data
                except MissingDocError:
                    # at training time we warn but ignore on missing docs
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )


from profane import import_all_modules

import_all_modules(__file__, __package__)
