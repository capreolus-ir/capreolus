import torch

from torch import nn

from capreolus.reranker.reranker import Reranker
from capreolus.extractor.embedtext import EmbedText
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix, RbfKernel, RbfKernelBank
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class KNRM_class(nn.Module):
    # based on CedrKnrmRanker from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, embeddings, config):
        super(KNRM_class, self).__init__()
        pad_token = config["pad_token"]
        self.p = config

        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = RbfKernelBank(mus, sigmas, dim=1, requires_grad=config["gradkernels"])

        self.embedding = create_emb_layer(embeddings, non_trainable=True)
        self.simmat = SimilarityMatrix(padding=pad_token)

        channels = 1
        if config["singlefc"]:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 1)]
        else:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 30), nn.Tanh(), nn.Linear(30, 1)]
        if config["scoretanh"]:
            combine_steps.append(nn.Tanh())
        self.combine = nn.Sequential(*combine_steps)

    def forward(self, doctoks, querytoks, query_idf):
        doc = self.embedding(doctoks)
        query = self.embedding(querytoks)
        simmat = self.simmat(query, doc, querytoks, doctoks)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = (
            simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN)
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN)
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        )
        result = kernels.sum(dim=3)  # sum over document
        mask = simmat.sum(dim=3) != 0.0  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms
        scores = self.combine(result)  # linear combination over kernels
        return scores


dtype = torch.FloatTensor


@Reranker.register
class KNRM(Reranker):
    description = """Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR'17."""
    EXTRACTORS = [EmbedText]

    @classmethod
    def get_model_class(cls):
        return KNRM_class

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        return locals().copy()  # ignored by sacred

    @staticmethod
    def required_params():
        # Used for validation. Returns a set of params required by the class defined in get_model_class()
        return {"pad_token", "gradkernels", "singlefc", "scoretanh"}

    def build(self):
        config = self.config.copy()
        config["pad_token"] = EmbedText.pad
        self.model = KNRM_class(self.embeddings, config)
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
