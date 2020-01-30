from capreolus.reranker.reranker import Reranker
from capreolus.extractor.embedtext import EmbedText
import torch
from torch import nn
from torch.nn import functional as F
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix


# TODO add shuffle, cascade, disambig?
class PACRR_class(nn.Module):
    # based on CedrPacrrRanker from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    @classmethod
    def alternate_init(cls, embedding, config):
        return cls(embedding, config)

    def __init__(self, weights_matrix, config):
        super(PACRR_class, self).__init__()
        p = config
        self.p = p

        self.embedding_dim = weights_matrix.shape[1]
        self.embedding = create_emb_layer(weights_matrix, non_trainable=True)
        self.simmat = SimilarityMatrix(padding=config["pad_token"])

        self.ngrams = nn.ModuleList()
        for ng in range(p["mingram"], p["maxgram"] + 1):
            self.ngrams.append(PACRRConvMax2dModule(ng, p["nfilters"], k=p["kmax"], channels=1))

        qterm_size = len(self.ngrams) * p["kmax"] + (1 if p["idf"] else 0)
        self.linear1 = torch.nn.Linear(p["maxqlen"] * qterm_size, p["combine"])
        self.linear2 = torch.nn.Linear(p["combine"], p["combine"])
        self.linear3 = torch.nn.Linear(p["combine"], 1)

        if p["nonlinearity"] == "none":
            nonlinearity = torch.nn.Identity
        elif p["nonlinearity"] == "relu":
            nonlinearity = torch.nn.ReLU
        elif p["nonlinearity"] == "tanh":
            nonlinearity = torch.nn.Tanh

        self.combine = torch.nn.Sequential(self.linear1, nonlinearity(), self.linear2, nonlinearity(), self.linear3)

    def forward(self, sentence, query_sentence, query_idf):
        doc = self.embedding(sentence)
        query = self.embedding(query_sentence)
        simmat = self.simmat(query, doc, query_sentence, sentence)

        scores = [ng(simmat) for ng in self.ngrams]
        if self.p["idf"]:
            scores.append(F.softmax(query_idf.reshape(query_idf.shape, 1).float(), dim=1).view(-1, self.p["maxqlen"], 1))
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


@Reranker.register
class PACRR(Reranker):
    description = """Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2017. PACRR: A Position-Aware Neural IR Model for Relevance Matching. In EMNLP'17."""
    EXTRACTORS = [EmbedText]

    @staticmethod
    def config():
        mingram = 1  # minimum length of ngram used
        maxgram = 3  # maximum length of ngram used
        nfilters = 32  # number of filters in convolution layer
        idf = True  # concatenate idf signals to combine relevance score from individual query terms
        kmax = 2  # value of kmax pooling used
        combine = 32  # size of combination layers
        nonlinearity = "relu"
        return locals().copy()  # ignored by sacred

    @staticmethod
    def required_params():
        # Used for validation. Returns a set of params required by the class defined in get_model_class()
        return {"mingram", "maxgram", "nfilters", "batch", "maxqlen", "idf", "kmax"}

    @classmethod
    def get_model_class(cls):
        return PACRR_class

    def build(self):
        config = self.config.copy()
        config["pad_token"] = EmbedText.pad
        self.model = PACRR_class(self.embeddings, config)
        return self.model

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]

    def test(self, query_sentence, query_idf, pos_sentence, *args, **kwargs):
        return self.model(pos_sentence, query_sentence, query_idf).view(-1)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)
