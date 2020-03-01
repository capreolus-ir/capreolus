import torch
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from torch import nn
from reranker import Reranker
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix, RbfKernelBank

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class CustomKNRM(nn.Module):
    # based on CedrKnrmRanker from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, extractor, config):
        super(CustomKNRM, self).__init__()
        self.p = config

        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = RbfKernelBank(mus, sigmas, dim=1, requires_grad=config["gradkernels"])

        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
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
        """
        Return vector embeddings (usually glove6b) for each token
        """
        return self.embedding(toks)

    def forward(self, doctoks, querytoks, query_idf):
        doc = self.get_embedding(doctoks)
        query = self.get_embedding(querytoks)
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


class TK_class(CustomKNRM):
    '''
    Adapted from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    '''

    def __init__(self, extractor, config):
        super(TK_class, self).__init__(extractor, config)
        self.embeddim = extractor.embeddings.shape[1]  # Dim of the glove6b embedding
        self.attention_encoder = StackedSelfAttentionEncoder(input_dim=self.embeddim,
                                                          hidden_dim=self.embeddim,
                                                          projection_dim=config["projdim"],
                                                          feedforward_hidden_dim=config["ffdim"],
                                                          num_layers=config["numlayers"],
                                                          num_attention_heads=config["numattheads"],
                                                          dropout_prob=0,
                                                          residual_dropout_prob=0,
                                                          attention_dropout_prob=0)
        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.5, dtype=torch.float32, requires_grad=True))

    def get_embedding(self, toks):
        """
        Overrides KNRM_Class's get_embedding to return contextualized word embeddings
        """
        embedding = self.embedding(toks)
        # TODO: Hoffstaeter's implementation makes use of masking. Check if it's required here
        # See https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py#L88
        contextual_embedding = self.attention_encoder(embedding, None)

        return self.mixer * embedding + (1 - self.mixer) * contextual_embedding


class TK(Reranker):
    name = "TK"
    citation = """Add citation"""
    # TODO: Declare the dependency on EmbedText

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        projdim = 32
        ffdim = 100
        numlayers = 2
        numattheads = 16

    def build(self):
        if not hasattr(self, "model"):
            self.model = TK_class(self["extractor"], self.cfg)
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

    def query(self, query, docids):
        if not hasattr(self["extractor"], "docid2toks"):
            raise RuntimeError(
                "reranker's extractor has not been created yet. try running the task's train() method first.")

        results = []
        for docid in docids:
            d = self["extractor"].id2vec(qid=None, query=query, posid=docid)
            results.append(self.test(d))
        return results
