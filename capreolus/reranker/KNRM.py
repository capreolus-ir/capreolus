import matplotlib.pyplot as plt
import torch
from torch import nn

from capreolus import ConfigOption
from capreolus.reranker import Reranker
from capreolus.reranker.common import RbfKernelBank, SimilarityMatrix, create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class KNRM_class(nn.Module):
    # based on CedrKnrmRanker from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, extractor, config):
        super(KNRM_class, self).__init__()
        self.p = config

        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = RbfKernelBank(mus, sigmas, dim=1, requires_grad=config["gradkernels"])
        non_trainable = not self.p["finetune"]
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=non_trainable)
        self.simmat = SimilarityMatrix(self.embedding)

        channels = 1
        if config["singlefc"]:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 1)]
        else:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 30), nn.Tanh(), nn.Linear(30, 1)]
        if config["scoretanh"]:
            combine_steps.append(nn.Tanh())
        self.combine = nn.Sequential(*combine_steps)

    def get_embedding(self, toks):
        return self.embedding(toks)

    def forward(self, doctoks, querytoks, query_idf):
        simmat = self.simmat(querytoks, doctoks)
        kernels = self.kernels(simmat)
        VIEWS = 1
        BATCH, KERNELS, QLEN, DLEN = kernels.shape
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


@Reranker.register
class KNRM(Reranker):
    """Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR'17."""

    module_name = "KNRM"

    config_spec = [
        ConfigOption("gradkernels", True, "backprop through mus and sigmas"),
        ConfigOption("scoretanh", False, "use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)"),
        ConfigOption("singlefc", True, "use single fully connected layer as in paper (True) or 2 fully connected layers (False)"),
        ConfigOption("finetune", False, "fine tune the embedding layer"),  # TODO check save when True
    ]

    def add_summary(self, summary_writer, niter):
        super(KNRM, self).add_summary(summary_writer, niter)
        if self.config["singlefc"]:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(self.model.combine[0].weight.data.cpu())
            summary_writer.add_figure("combine_steps weight", fig, niter)
        else:
            pass

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = KNRM_class(self.extractor, self.config)

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
