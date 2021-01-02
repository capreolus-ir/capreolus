from torch.nn import CosineSimilarity
import torch.nn.functional as F
from collections import defaultdict
from capreolus import Dependency, ModuleBase, get_logger
from capreolus.sampler import TrainTripletSampler

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
                
    def train_encoder(self):
        train_dataset = TrainTripletSampler()
        train_run = self.create_fake_train_run()
        docids = [docid for docid_list in train_run.values() for docid in docid_list]
        self.extractor.preprocess(
            qids=train_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type]
        )

        train_run = {qid: docids for qid, docids in train_run.items() if qid in self.benchmark.folds["s1"]["train_qids"]}
        train_dataset.prepare(train_run, self.benchmark.qrels, self.extractor, relevance_level=self.benchmark.relevance_level)

        dev_dataset = TrainTripletSampler()
        train_run = {qid: docids for qid, docids in train_run.items() if qid in self.benchmark.folds["s1"]["predict"]["dev"]}
        dev_dataset.prepare(train_run, self.benchmark.qrels, self.extractor, relevance_level=self.benchmark.relevance_level)

        self.build_model()
        self.trainer.train(self, train_dataset, dev_dataset)
        
    def build_model(self):
        """
        Initialize the PyTorch model
        """
        raise NotImplementedError
    
    def encode(self, numericalized_text):
        return self.model(numericalized_text)

    def score(self, d):
        query = d["query"]
        posdoc = d["posdoc"]
        negdoc = d["negdoc"]
        
        query_emb = self.model(query)
        posdoc_emb = self.model(posdoc)
        negdoc_emb = self.model(negdoc)

        return [F.cosine_similarity(query_emb, posdoc_emb), F.cosine_similarity(query_emb, negdoc_emb)]
            

from profane import import_all_modules

from .TinyBERT import TinyBERTEncoder


import_all_modules(__file__, __package__)
