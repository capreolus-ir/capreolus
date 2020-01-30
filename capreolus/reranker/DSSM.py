from capreolus.extractor.bagofwords import BagOfWords
from capreolus.extractor.embedtext import EmbedText
from capreolus.reranker.reranker import Reranker

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM_class(nn.Module):
    def __init__(self, stoi, p):
        super(DSSM_class, self).__init__()
        self.p = p
        nvocab = len(stoi)
        nhiddens = [nvocab] + list(map(int, p["nhiddens"].split()))
        print(nhiddens)

        self.ffw = nn.Sequential()
        for i in range(len(nhiddens) - 1):
            self.ffw.add_module("linear%d" % i, nn.Linear(nhiddens[i], nhiddens[i + 1]))
            self.ffw.add_module("activate%d" % i, nn.ReLU())
            self.ffw.add_module("dropout%i" % i, nn.Dropout(0.5))

        self.output_layer = nn.Sigmoid()

    def forward(self, sentence, query):
        query = self.ffw(query)
        sentence = self.ffw(sentence)

        query_norm = query.norm(dim=-1)[:, None] + 1e-7
        sentence_norm = sentence.norm(dim=-1)[:, None] + 1e-7

        query = query / query_norm
        sentence = sentence / sentence_norm

        cos_x = (query * sentence).sum(dim=-1, keepdim=True)

        score = self.output_layer(cos_x)
        return score


dtype = torch.FloatTensor


@Reranker.register
class DSSM(Reranker):
    description = """Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In CIKM'13."""
    EXTRACTORS = [BagOfWords]

    @staticmethod
    def config():
        # hidden layer dimentions, should be a list of space-separated number in a string, e.g. '56 128 32', the i-th value represents the output dim of the i-th hidden layer
        nhiddens = "56"
        lr = 0.0001
        return locals().copy()  # ignored by sacred

    @staticmethod
    def required_params():
        # Used for validation. Returns a set of params required by the class defined in get_model_class()
        return {"nhiddens", "nvocab", "maxdoclen", "maxqlen"}

    @classmethod
    def get_model_class(cls):
        return DSSM_class

    def build(self):
        self.model = DSSM_class(self.embeddings, self.config)
        return self.model

    def score(self, data):
        query_idf = data["query_idf"].to(self.device)
        query_sentence = data["query"].to(self.device)
        pos_sentence, neg_sentence = data["posdoc"].to(self.device), data["negdoc"].to(self.device)

        return [self.model(pos_sentence, query_sentence).view(-1), self.model(neg_sentence, query_sentence).view(-1)]

    def test(self, query_sentence, query_idf, pos_sentence, *args, **kwargs):
        query_sentence = query_sentence.to(self.device)
        pos_sentence = pos_sentence.to(self.device)

        return self.model(pos_sentence, query_sentence).view(-1)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)
