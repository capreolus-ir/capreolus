import torch
import tensorflow as tf
from tensorflow.keras.layers import Layer
from torch import nn

from capreolus.reranker import Reranker, TensorFlowReranker
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix, RbfKernel, RbfKernelBank
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


class KNRM_TF_Class(Layer):
    def __init__(self, extractor, config, **kwargs):
        self.config = config
        self.extractor = extractor
        self.embedding = None
        super(KNRM_TF_Class, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(len(self.extractor.stoi), self.extractor.embeddings.shape[1], weights=[self.extractor.embeddings], trainable=False)

    def call(self, input, **kwargs):
        """
        All the inputs are arrays of indices into an embedding matrix
        """
        posdoc, query, query_idf = input[0], input[1], input[2]
        query_embed = self.embedding(query)
        print("query is {}".format(query))
        print("query embed is {}".format(query_embed))
        return 1

    def compute_output_shape(self, input_shape):
        return 1


class KNRM_TF(TensorFlowReranker):
    name = "KNRM_TF"

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        finetune = False  # Fine tune the embedding

    def build(self):
        self.model = KNRM_TF_Class(self["extractor"], self.cfg)
        return self.model

    def score(self, posdoc, negdoc, query, query_idf):
        return [
            self.model(posdoc, query, query_idf),
            self.model(negdoc, query, query_idf)
        ]

    def test(self, doc, query, query_idf):
        return self.model(doc, query, query_idf)


class KNRM(Reranker):
    name = "KNRM"
    citation = """Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017.
                  End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR'17."""

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a    nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        finetune = False  # Fine tune the embedding

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


