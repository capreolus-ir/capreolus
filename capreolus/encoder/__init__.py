import torch.nn.functional as F
import torch
import pickle
import os
from collections import defaultdict
from capreolus import Dependency, ModuleBase, get_logger, constants
from capreolus.sampler import TrainTripletSampler, PredSampler

logger = get_logger(__name__)


class Encoder(ModuleBase):
    """
    Base class for encoders. Encoders take a document and convert it into a vector, and the result is usually put in a FAISS index for an approximate nearest-neighbour search
    """

    module_type = "encoder"
    dependencies = [
        Dependency(key="trainer", module="trainer", name="pytorchann"),
        Dependency(key="sampler", module="sampler", name="triplet"),
        Dependency(key="extractor", module="extractor", name="berttext"),
        Dependency(key="benchmark", module="benchmark")
    ]

    def get_results_path(self):
        """Return an absolute path that can be used for storing results.
        The path is a function of the module's config and the configs of its dependencies.
        """

        return constants["RESULTS_BASE_PATH"] / self.get_module_path()
    
    def exists(self, weights_fn):
        return os.path.isfile(weights_fn)

    def save_weights(self, weights_fn, optimizer):
        if not os.path.exists(os.path.dirname(weights_fn)):
            os.makedirs(os.path.dirname(weights_fn))

        d = {k: v for k, v in self.model.state_dict().items() if ("embedding.weight" not in k and "_nosave_" not in k)}
        with open(weights_fn, "wb") as outf:
            pickle.dump(d, outf, protocol=-1)

        optimizer_fn = weights_fn.as_posix() + ".optimizer"
        with open(optimizer_fn, "wb") as outf:
            pickle.dump(optimizer.state_dict(), outf, protocol=-1)

    def load_weights(self, weights_fn, optimizer):
        with open(weights_fn, "rb") as f:
            d = pickle.load(f)

        cur_keys = set(k for k in self.model.state_dict().keys() if not ("embedding.weight" in k or "_nosave_" in k))
        missing = cur_keys - set(d.keys())
        if len(missing) > 0:
            raise RuntimeError("loading state_dict with keys that do not match current model: %s" % missing)

        self.model.load_state_dict(d, strict=False)

        optimizer_fn = weights_fn.as_posix() + ".optimizer"
        with open(optimizer_fn, "rb") as f:
            optimizer.load_state_dict(pickle.load(f))

    def create_fake_train_run(self):
        train_run = defaultdict(list)
        for qid, docid_to_label in self.benchmark.qrels.items():
            # TODO: Do not hard-code s1 here
            if qid not in self.benchmark.folds["s1"]["train_qids"]:
                continue

            for docid, label in docid_to_label.items():
                # 0 is the "score" here. Doesn't matter what this value is since we are not going to use te score
                train_run[qid].append(docid)

        return train_run

    def create_fake_dev_run(self):
        dev_run = defaultdict(list)
        for qid, docid_to_label in self.benchmark.qrels.items():
            # TODO: Do not hard-code s1 here
            if qid not in self.benchmark.folds["s1"]["predict"]["dev"]:
                continue

            for docid, label in docid_to_label.items():
                # 0 is the "score" here. Doesn't matter what this value is since we are not going to use te score
                dev_run[qid].append(docid)

        return dev_run
                
    def train_encoder(self, train_run, dev_run, docids, qids):
        train_dataset = self.sampler
        self.extractor.preprocess(
            qids=qids, docids=docids, topics=self.benchmark.topics
        )

        train_dataset.prepare(train_run, self.benchmark.qrels, self.extractor, relevance_level=self.benchmark.relevance_level)

        dev_dataset = PredSampler()
        dev_dataset.prepare(dev_run, self.benchmark.qrels, self.extractor, relevance_level=self.benchmark.relevance_level)

        self.instantiate_model()
        output_path = self.get_results_path()
        self.trainer.train(self, train_dataset, dev_dataset, output_path, self.benchmark.qrels)

    def build_model(self, train_run, dev_run, docids, qids):
        self.train_encoder(train_run, dev_run, docids, qids)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
    def instantiate_model(self):
        """
        Initialize the PyTorch model
        """
        raise NotImplementedError

    def encode_doc(self, numericalized_text, mask):
        """
        Override this if an encode needs to treat queries and documents differently (eg: RepBERT)
        """
        return self.encode(numericalized_text, mask)

    def encode_query(self, numericalized_text, mask):
        return self.encode(numericalized_text, mask)

    def encode(self, numericalized_text, mask):
        return self.model(numericalized_text, mask)

    def score(self, d):
        query = d["query"]
        query_mask = d["query_mask"]
        posdoc = d["posdoc"]
        posdoc_mask = d["posdoc_mask"]
        negdoc = d["negdoc"]
        negdoc_mask = d["negdoc_mask"]
        
        query_emb = self.model(query, query_mask)
        posdoc_emb = self.model(posdoc, posdoc_mask)
        negdoc_emb = self.model(negdoc, negdoc_mask)

        return [F.cosine_similarity(query_emb, posdoc_emb), F.cosine_similarity(query_emb, negdoc_emb)]

    def test(self, d):
        query = d["query"]
        query_mask = d["query_mask"]
        doc = d["posdoc"]
        doc_mask = d["posdoc_mask"]

        query_emb = self.model(query, query_mask)
        doc_emb = self.model(doc, doc_mask)
        
        return F.cosine_similarity(query_emb, doc_emb)
            

from profane import import_all_modules

from .SentenceBERT import SentenceBERTEncoder
from .CLEAR import CLEAREncoder


import_all_modules(__file__, __package__)
