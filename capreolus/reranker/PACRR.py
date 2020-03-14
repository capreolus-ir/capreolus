import torch
from torch import nn
from torch.nn import functional as F


# TODO add shuffle, cascade, disambig?
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix
from capreolus.reranker import Reranker


class PACRR_class(nn.Module):
    # based on CedrPacrrRanker from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license

    def __init__(self, extractor, config):
        super(PACRR_class, self).__init__()
        p = config
        self.p = p
        self.extractor = extractor
        self.embedding_dim = extractor.embeddings.shape[1]
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        self.simmat = SimilarityMatrix(padding=extractor.pad)

        self.ngrams = nn.ModuleList()
        for ng in range(p["mingram"], p["maxgram"] + 1):
            self.ngrams.append(
                PACRRConvMax2dModule(ng, p["nfilters"], k=p["kmax"], channels=1)
            )

        qterm_size = len(self.ngrams) * p["kmax"] + (1 if p["idf"] else 0)
        self.linear1 = torch.nn.Linear(
            extractor.cfg["maxqlen"] * qterm_size, p["combine"]
        )
        self.linear2 = torch.nn.Linear(p["combine"], p["combine"])
        self.linear3 = torch.nn.Linear(p["combine"], 1)

        if p["nonlinearity"] == "none":
            nonlinearity = torch.nn.Identity
        elif p["nonlinearity"] == "relu":
            nonlinearity = torch.nn.ReLU
        elif p["nonlinearity"] == "tanh":
            nonlinearity = torch.nn.Tanh

        self.combine = torch.nn.Sequential(
            self.linear1, nonlinearity(), self.linear2, nonlinearity(), self.linear3
        )

    def forward(self, sentence, query_sentence, query_idf):
        doc = self.embedding(sentence)
        query = self.embedding(query_sentence)
        simmat = self.simmat(query, doc, query_sentence, sentence)

        scores = [ng(simmat) for ng in self.ngrams]
        if self.p["idf"]:
            scores.append(
                F.softmax(query_idf.reshape(query_idf.shape, 1).float(), dim=1).view(
                    -1, self.extractor.cfg["maxqlen"], 1
                )
            )
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        rel = self.combine(scores)
        return rel


class PACRRConvMax2dModule(torch.nn.Module):
    # based on PACRRConvMax2dModule from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, shape, n_filters, k, channels):
        super().__init__()
        self.shape = shape
        if shape != 1:
            self.pad = torch.nn.ConstantPad2d((0, shape - 1, 0, shape - 1), 0)
        else:
            self.pad = None
        self.conv = torch.nn.Conv2d(channels, n_filters, shape)
        self.activation = torch.nn.ReLU()
        self.k = k
        self.shape = shape
        self.channels = channels

    def forward(self, simmat):
        BATCH, CHANNELS, QLEN, DLEN = simmat.shape
        if self.pad:
            simmat = self.pad(simmat)
        conv = self.activation(self.conv(simmat))
        top_filters, _ = conv.max(dim=1)
        top_toks, _ = top_filters.topk(self.k, dim=2)
        result = top_toks.reshape(BATCH, QLEN, self.k)
        return result


class PACRR(Reranker):
    name = "PACRR"
    citation = "Kai Hui1, Andrew Yates1, Klaus Berberich1, Gerard de Melo, EMNLP 2017"

    @staticmethod
    def config():
        mingram = 1  # minimum length of ngram used
        maxgram = 3  # maximum length of ngram used
        nfilters = 32  # number of filters in convolution layer
        idf = (
            True
        )  # concatenate idf signals to combine relevance score from individual query terms
        kmax = 2  # value of kmax pooling used
        combine = 32  # size of combination layers
        nonlinearity = (
            "relu"
        )  # nonlinearity in combination layer: 'none', 'relu', 'tanh'

    # TODO: Move to a common place
    def build(self):
        if not hasattr(self, "model"):
            self.model = PACRR_class(self["extractor"], self.cfg)
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
