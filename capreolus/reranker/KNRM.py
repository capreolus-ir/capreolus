import torch
import tensorflow as tf
from torch import nn
import matplotlib.pyplot as plt

from capreolus.reranker import PyTorchReranker, TensorFlowReranker
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix, RbfKernel, RbfKernelBank
from capreolus.utils.loginit import get_logger
from capreolus.reranker.common import RbfKernelBankTF, SimilarityMatrixTF
from capreolus.reranker.common import alternate_simmat, RbfKernelTF

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
        self.simmat = SimilarityMatrix(padding=extractor.pad)

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
        doc = self.get_embedding(doctoks)
        query = self.get_embedding(querytoks)

        # query = torch.rand_like(query)  # debug
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


class KNRM_TF_Class(tf.keras.Model):
    def __init__(self, extractor, config, **kwargs):
        super(KNRM_TF_Class, self).__init__(**kwargs)
        self.config = config
        self.extractor = extractor
        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        # self.embedding = tf.keras.layers.Embedding(len(self.extractor.stoi), self.extractor.embeddings.shape[1], weights=[self.extractor.embeddings], trainable=False)
        self.embedding = tf.keras.layers.Embedding(len(self.extractor.stoi), self.extractor.embeddings.shape[1], trainable=True)
        self.kernels = RbfKernelBankTF(mus, sigmas, dim=1, requires_grad=True)
        self.simmat = SimilarityMatrixTF(padding=extractor.pad)
        self.combine = tf.keras.layers.Dense(1, input_shape=(self.kernels.count(),))
        # self.dummy_combine = tf.keras.layers.Dense(1, input_shape=(11, extractor.cfg["maxqlen"], extractor.cfg["maxdoclen"],))

        # Flags to make sure that tf.Variable gets called in call() only once.
        # See this: https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#functions-that-create-state
        self.is_simmat_var = False
        self.is_kernel_var = False
        self.is_score_var = False

    def get_score(self, doc, query, query_idf):
        query = self.embedding(query)
        doc = self.embedding(doc)
        simmat = alternate_simmat(query, doc)

        k = self.kernels(simmat)
        doc_k = tf.reduce_sum(k, axis=3)  # sum over document
        log_k = tf.math.log(doc_k + 1e-6)
        query_k = tf.reduce_sum(log_k, axis=2)
        # TODO: Fix/handle padding tokens
        scores = self.combine(query_k)

        return scores

        # return self.dummy_combine(simmat)

    @tf.function
    def call(self, x, **kwargs):
        posdoc, negdoc, query, query_idf = x[0], x[1], x[2], x[3]
        posdoc_score, negdoc_score = self.get_score(posdoc, query, query_idf), self.get_score(negdoc, query, query_idf)

        # During eval, the negdoc_score would be a zero tensor
        # TODO: Verify that negdoc_score is indeed always zero whenever a zero negdoc tensor is passed into it
        return posdoc_score - negdoc_score


class KNRM_TF(TensorFlowReranker):
    name = "KNRMTF"

    @staticmethod
    def config():
        gradkernels = True # backprop through mus and sigmas
        finetune = False  # Fine tune the embedding

    def build(self):
        self.model = KNRM_TF_Class(self["extractor"], self.cfg)
        return self.model

    def score(self, posdoc, negdoc, query, query_idf):
        return [
            self.model((posdoc, query, query_idf)),
            self.model((negdoc, query, query_idf))
        ]

    def test(self, doc, query, query_idf):
        return self.model((doc, query, query_idf))


class KNRM(PyTorchReranker):
    name = "KNRM"
    citation = """Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017.
                  End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR'17."""

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a    nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        finetune = False  # Fine tune the embedding

    def add_summary(self, summary_writer, niter):
        super(KNRM, self).add_summary(summary_writer, niter)
        if self.cfg["singlefc"]:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(self.model.combine[0].weight.data.cpu())
            summary_writer.add_figure("combine_steps weight", fig, niter)
        else:
            pass

    def build(self):
        self.model = KNRM_class(self["extractor"], self.cfg)

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


