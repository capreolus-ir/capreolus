import torch.nn.functional as F
from torch.nn import DataParallel
from capreolus import Dependency
from capreolus.encoder import Encoder
from capreolus.encoder.SentenceBERT import SentenceBERTEncoder_Class


@Encoder.register
class CLEAREncoder(Encoder):
    module_name = "clear"
    dependencies = [
        Dependency(key="trainer", module="trainer", name="pytorchann"),
        Dependency(key="sampler", module="sampler", name="residualtriplet"),
        Dependency(key="extractor", module="extractor", name="berttext"),
        Dependency(key="benchmark", module="benchmark")
    ]

    def instantiate_model(self):
        if not hasattr(self, "model"):
            self.model = DataParallel(SentenceBERTEncoder_Class())
            self.hidden_size = self.model.module.hidden_size

        return self.model

    def score(self, d):
        residual = d["residual"]
        query = d["query"]
        query_mask = d["query_mask"]
        posdoc = d["posdoc"]
        posdoc_mask = d["posdoc_mask"]
        negdoc = d["negdoc"]
        negdoc_mask = d["negdoc_mask"]

        query_emb = self.model(query, query_mask)
        posdoc_emb = self.model(posdoc, posdoc_mask)
        negdoc_emb = self.model(negdoc, negdoc_mask)

        # Ok so the loss function that is applied should be [residual - cos(q, posdoc) + cos(q, negdoc)]
        # Instead of implementing ^ this loss function, we are going to try to write it in the standard pairwise
        # loss function form:
        # [residual - cos(q, posdoc) + cos(q, negdoc)] = needs to be written in the form (1 - x + y)
        # = [1 - 1 + residual - cos(q, posdoc) + cos(q, negdoc)]
        # = [1 - [1 - residual + cos(q, posdoc)] + cos(q, negdoc)]
        #  ^ This is in the required form. Hence if we change the score from the posdoc a little bit, we can use the
        # standard pairwise hinge function. Saves as an annoying if else condition in pytorchtrainer :)

        posdoc_score = 1 - residual + F.cosine_similarity(query_emb, posdoc_emb)
        negdoc_score = F.cosine_similarity(query_emb, negdoc_emb)

        return [posdoc_score, negdoc_score]
