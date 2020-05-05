import torch
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow_core.python import name_scope

_hinge_loss = torch.nn.MarginRankingLoss(margin=1, reduction="mean")


def pair_softmax_loss(pos_neg_scores):
    scores = torch.stack(pos_neg_scores, dim=1)
    return torch.mean(1.0 - scores.softmax(dim=1)[:, 0])


def pair_hinge_loss(pos_neg_scores):
    label = torch.ones_like(pos_neg_scores[0])  # , dtype=torch.int)
    return _hinge_loss(pos_neg_scores[0], pos_neg_scores[1], label)


def tf_pair_hinge_loss(labels, scores):
    """
    Labels - a dummy zero tensor.
    Scores - A tensor of the shape (batch_size, diff), where diff = posdoc_score - negdoc_score
    """
    ones = tf.ones_like(scores)
    zeros = tf.ones_like(scores)

    return K.sum(K.maximum(zeros, ones - scores))


def alternate_simmat(query_embed, doc_embed):
    """
    The shape of input tensor is (maxdoclen, embeddim,
    """
    # assert query_embed.shape[0] == doc_embed.shape[0]
    batch_size, qlen, doclen = tf.shape(query_embed)[0], tf.shape(query_embed)[1], tf.shape(doc_embed)[1]
    # print("Batch size is {0} qlen is {1} and doclen is {2}".format(batch_size, qlen, doclen))
    # qlen, doclen = tf.shape(query_embed)[1], tf.shape(doc_embed)[1]
    q_denom = tf.broadcast_to(tf.reshape(tf.norm(query_embed, axis=2), (batch_size, qlen, 1)),
                              (batch_size, qlen, doclen,)) + 1e-9
    doc_denom = tf.broadcast_to(tf.reshape(tf.norm(doc_embed, axis=2), (batch_size, 1, doclen)),
                                (batch_size, qlen, doclen,)) + 1e-9

    # Why perm?
    # let query have shape (32, 8, 300)
    # let doc have shape (32, 800, 300)
    # Our similarity matrix should have the shape (32, 8, 800)
    # The perm is required so that the result of matmul will have this shape
    perm = tf.transpose(doc_embed, perm=[0, 2, 1])
    sim = tf.matmul(query_embed, perm) / (q_denom * doc_denom)

    # TODO: Add support for handling list inputs (eg: for CEDR). See the pytorch implementation of simmat
    return sim


class SimilarityMatrixTF(Layer):
    """
    Takes in a list of query tokens and doc tokens (and their embeddings) and returns
    a cosine similarity matrix
    """
    def __init__(self, padding=0, **kwargs):
        super(SimilarityMatrixTF, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs, **kwargs):
        query_embed, doc_embed, query_tok, doc_tok = inputs[0], inputs[1], inputs[2], inputs[3]
        assert query_embed.shape[0] == doc_embed.shape[0]
        batch_size, qlen, doclen = query_embed.shape[0], query_embed.shape[1],  doc_embed.shape[1]
        q_denom = tf.broadcast_to(tf.reshape(tf.norm(query_embed, axis=2), (batch_size, qlen, 1)), (batch_size, qlen, doclen)) + 1e-9
        doc_denom = tf.broadcast_to(tf.reshape(tf.norm(doc_embed, axis=2), (batch_size, 1, doclen)), (batch_size, qlen, doclen)) + 1e-9

        # Why perm?
        # let query have shape (32, 8, 300)
        # let doc have shape (32, 800, 300)
        # Our similarity matrix should have the shape (32, 8, 800)
        # The perm is required so that the result of matmul will have this shape
        perm = tf.transpose(doc_embed, perm=[0, 2, 1])
        sim = tf.matmul(query_embed, perm) / (q_denom * doc_denom)

        nul = tf.zeros_like(sim)
        sim = tf.where(tf.broadcast_to(tf.reshape(query_tok, (batch_size, qlen, 1)), (batch_size, qlen, doclen)) == self.padding, nul, sim)
        sim = tf.where(tf.broadcast_to(tf.reshape(doc_tok, (batch_size, 1, doclen)), (batch_size, qlen, doclen)) == self.padding, nul, sim)

        # TODO: Add support for handling list inputs (eg: for CEDR). See the pytorch implementation of simmat
        return tf.stack([sim], axis=1)


class SimilarityMatrix(torch.nn.Module):
    # based on SimmatModule from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding

    # query_embed and doc_embed can be a list (eg for CEDR)
    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []

        assert type(query_embed) == type(doc_embed)
        if not isinstance(query_embed, list):
            query_embed, doc_embed = [query_embed], [doc_embed]

        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]

            if a_emb is None and b_emb is None:
                # exact match matrix
                sim = query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == doc_tok.reshape(BAT, 1, B).expand(BAT, A, B).float()
            else:
                # cosine similarity matrix
                a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9  # avoid 0div
                b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9  # avoid 0div
                perm = b_emb.permute(0, 2, 1)
                sim = a_emb.bmm(perm) / (a_denom * b_denom)

            # set similarity values to 0 for <pad> tokens in query and doc (indicated by self.padding)
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)

            simmat.append(sim)
        return torch.stack(simmat, dim=1)


class RbfKernel(torch.nn.Module):
    # based on KNRMRbfKernel from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = torch.nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


class RbfKernelBank(torch.nn.Module):
    # based on KNRMRbfKernelBank from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [RbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = torch.nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class RbfKernelBankTF(Layer):
    def __init__(self, mus, sigmas, dim=1, requires_grad=True, **kwargs):
        super(RbfKernelBankTF, self).__init__(**kwargs)
        self.dim = dim
        self.kernel_list = [RbfKernelTF(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]

    def count(self):
        return len(self.kernel_list)

    def call(self, data, **kwargs):
        return tf.stack([self.kernel_list[i](data) for i in range(len(self.kernel_list))], axis=self.dim)


class RbfKernelTF(Layer):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True, **kwargs):
        super(RbfKernelTF, self).__init__(**kwargs)
        self.mu = tf.Variable(initial_mu, trainable=requires_grad, name="mus", dtype=tf.float32)
        self.sigma = tf.Variable(initial_sigma, trainable=requires_grad, name="sigmas", dtype=tf.float32)

    def call(self, data, *kwargs):
        adj = data - self.mu
        return tf.exp(-0.5 * adj * adj / self.sigma / self.sigma)


def create_emb_layer(weights, non_trainable=True):
    layer = torch.nn.Embedding(*weights.shape)
    layer.load_state_dict({"weight": torch.tensor(weights)})

    if non_trainable:
        layer.weight.requires_grad = False
    else:
        layer.weight.requires_grad = True

    return layer
