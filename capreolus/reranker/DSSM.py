from capreolus.extractor.bagofwords import BagOfWords

import torch
import torch.nn as nn

from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class DSSM_class(nn.Module):
    def __init__(self, extractor, config):
        super(DSSM_class, self).__init__()
        p = config
        nvocab = len(extractor.stoi)
        nhiddens = [nvocab] + list(map(int, p["nhiddens"].split()))
        self.ffw = nn.Sequential()
        for i in range(len(nhiddens) - 1):
            self.ffw.add_module("linear%d" % i, nn.Linear(nhiddens[i], nhiddens[i + 1]))
            self.ffw.add_module("activate%d" % i, nn.ReLU())
            self.ffw.add_module("dropout%i" % i, nn.Dropout(0.5))

        self.output_layer = nn.Sigmoid()

    def forward(self, sentence, query, query_idf):
        query = query.float()
        sentence = sentence.float()
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


class DSSM(Reranker):
    description = """Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In CIKM'13."""
    EXTRACTORS = [BagOfWords]
    name = "DSSM"

    @staticmethod
    def config():
        # hidden layer sizes, like '56 128', where i'th value indicates output size of the i'th layer
        nhiddens = "56"
        lr = 0.0001

    @classmethod
    def get_model_class(cls):
        return DSSM_class

    def build(self):
        self.model = DSSM_class(self["extractor"], self.cfg)
        return self.model

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]

    def test(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence = d["posdoc"]
        return self.model(pos_sentence, query_sentence, query_idf).view(-1)
