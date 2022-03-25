import tensorflow as tf
import torch
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow_ranking.python.keras.losses import PairwiseHingeLoss

_hinge_loss = torch.nn.MarginRankingLoss(margin=1, reduction="mean")


class KerasPairModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(KerasPairModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, x, **kwargs):
        score = self.model.score(x, **kwargs)
        score = tf.cast(score, tf.float32)
        return score

    def predict_step(self, data):
        return self.model.predict_step(data)


class KerasTripletModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(KerasTripletModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, x, **kwargs):
        pos_score, neg_score = self.model.score_pair(x, **kwargs)
        score = tf.stack([pos_score, neg_score], axis=1)
        score = tf.cast(score, tf.float32)

        return score

    def predict_step(self, data):
        return self.model.predict_step(data)


class KerasLCEModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(KerasLCEModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, x, **kwargs):
        pos_score = self.model.score(x, **kwargs)[:, 1]
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        negdoc_bert_input = tf.transpose(negdoc_bert_input, perm=[1, 0, 2])
        negdoc_bert_input = tf.cast(negdoc_bert_input, tf.int64)
        negdoc_mask = tf.transpose(negdoc_mask, perm=[1, 0, 2])
        negdoc_mask = tf.cast(negdoc_mask, tf.int64)
        negdoc_seg = tf.transpose(negdoc_seg, perm=[1, 0, 2])
        negdoc_seg = tf.cast(negdoc_seg, tf.int64)

        all_scores = [pos_score]

        for i in range(0, len(negdoc_bert_input)):
            neg_x = (negdoc_bert_input[i], negdoc_mask[i], negdoc_seg[i], negdoc_bert_input, negdoc_mask, negdoc_seg)
            all_scores.append(self.model.score(neg_x, **kwargs)[:, 1])

        score = tf.stack(all_scores, axis=1)
        score = tf.cast(score, tf.float32)

        return score

    def predict_step(self, data):
        return self.model.predict_step(data)


class TFPairwiseHingeLoss(PairwiseHingeLoss):
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, 2])
        y_pred = tf.reshape(y_pred, [-1, 2])

        return super(TFPairwiseHingeLoss, self).call(y_true, y_pred)


class TFCategoricalCrossEntropyLoss(CategoricalCrossentropy):
    def call(self, ytrue, ypred):
        tf.debugging.assert_equal(tf.shape(ytrue), tf.shape(ypred))

        return super(TFCategoricalCrossEntropyLoss, self).call(ytrue, ypred)


class TFLCELoss(CategoricalCrossentropy):
    def call(self, ytrue, ypred):
        # NIMATODO: why is this the same as TFCategoricalCrossEntropyLoss
        tf.debugging.assert_equal(tf.shape(ytrue), tf.shape(ypred))
        return super(TFLCELoss, self).call(ytrue, ypred)


def pair_softmax_loss(pos_neg_scores, *args, **kwargs):
    scores = torch.stack(pos_neg_scores, dim=1)
    return torch.mean(1.0 - scores.softmax(dim=1)[:, 0])


def pair_hinge_loss(pos_neg_scores, *args, **kwargs):
    label = torch.ones_like(pos_neg_scores[0])  # , dtype=torch.int)
    return _hinge_loss(pos_neg_scores[0], pos_neg_scores[1], label)


def new_similarity_matrix_tf(query_embed, doc_embed, query_tok, doc_tok, padding):
    batch, qlen, dims = query_embed.shape
    doclen = doc_embed.shape[1]

    query_embed = tf.reshape(tf.nn.l2_normalize(query_embed, axis=-1), [batch, qlen, 1, dims])
    query_padding = tf.reshape(tf.cast(query_tok != padding, query_embed.dtype), [batch, qlen, 1, 1])
    query_embed = query_embed * query_padding

    doc_embed = tf.reshape(tf.nn.l2_normalize(doc_embed, axis=-1), [batch, 1, doclen, dims])
    doc_padding = tf.reshape(tf.cast(doc_tok != padding, doc_embed.dtype), [batch, 1, doclen, 1])
    doc_embed = doc_embed * doc_padding

    simmat = tf.reduce_sum(query_embed * doc_embed, axis=-1, keepdims=True)
    return simmat


def similarity_matrix_tf(query_embed, doc_embed, query_tok, doc_tok, padding):
    """Original TF similarity matrix. May have issues with mixed precision. Use new_similarity_matrix_tf instead"""
    batch_size, qlen, doclen = tf.shape(query_embed)[0], tf.shape(query_embed)[1], tf.shape(doc_embed)[1]
    q_denom = tf.broadcast_to(tf.reshape(tf.norm(query_embed, axis=2), (batch_size, qlen, 1)), (batch_size, qlen, doclen)) + 1e-9
    doc_denom = (
        tf.broadcast_to(tf.reshape(tf.norm(doc_embed, axis=2), (batch_size, 1, doclen)), (batch_size, qlen, doclen)) + 1e-9
    )

    # Why perm?
    # let query have shape (32, 8, 300)
    # let doc have shape (32, 800, 300)
    # Our similarity matrix should have the shape (32, 8, 800)
    # The perm is required so that the result of matmul will have this shape
    perm = tf.transpose(doc_embed, perm=[0, 2, 1])
    sim = tf.matmul(query_embed, perm) / (q_denom * doc_denom)
    nul = tf.zeros_like(sim)
    sim = tf.where(tf.broadcast_to(tf.reshape(query_tok, (batch_size, qlen, 1)), (batch_size, qlen, doclen)) == padding, nul, sim)
    sim = tf.where(tf.broadcast_to(tf.reshape(doc_tok, (batch_size, 1, doclen)), (batch_size, qlen, doclen)) == padding, nul, sim)
    return sim


class SimilarityMatrix(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.padding = 0

    def remove_padding(self, sim, query_tok, doc_tok, BAT, A, B):
        nul = torch.zeros_like(sim)
        sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
        sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)
        return sim

    def exact_match_matrix(self, query_tok, doc_tok, BAT, A, B):
        sim = (query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == doc_tok.reshape(BAT, 1, B).expand(BAT, A, B)).float()
        sim = self.remove_padding(sim, query_tok, doc_tok, BAT, A, B)
        return sim

    def cosine_similarity_matrix(self, query_tok, doc_tok, BAT, A, B):
        a_emb, b_emb = self.embedding(query_tok), self.embedding(doc_tok)
        a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9  # avoid 0div
        b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9  # avoid 0div
        perm = b_emb.permute(0, 2, 1)
        sim = a_emb.bmm(perm) / (a_denom * b_denom)
        sim = self.remove_padding(sim, query_tok, doc_tok, BAT, A, B)
        return sim

    # query_tok and doc_tok should contain integers
    def forward(self, query_tok, doc_tok):
        BAT, A, B = query_tok.shape[0], query_tok.shape[1], doc_tok.shape[1]
        assert doc_tok.shape[0] == BAT

        # note: all OOV terms are given negative indices and 0 is padding
        # approach:
        # 1. we calculate an exact match matrix on OOV terms only, and set padding to 0
        # 2. we calculate a cosine sim matrix on in-vocab terms only, and set padding to 0
        # 3. we sum the two matrices
        exact_match = self.exact_match_matrix(query_tok.clamp(max=0), doc_tok.clamp(max=0), BAT, A, B)
        cos_matrix = self.cosine_similarity_matrix(query_tok.clamp(min=0), doc_tok.clamp(min=0), BAT, A, B)
        simmat = exact_match + cos_matrix
        return simmat


# TODO replace this with newer ONIR version?
# https://github.com/Georgetown-IR-Lab/OpenNIR/blob/ca14dfa5e7cfef3fbbb35efbb4e7df0f1fbde590/onir/modules/interaction_matrix.py#L27
class StackedSimilarityMatrix(torch.nn.Module):
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
        data = tf.cast(data, tf.float32)
        adj = data - self.mu
        score = tf.exp(-0.5 * adj * adj / self.sigma / self.sigma)
        return score


def create_emb_layer(weights, non_trainable=True):
    layer = torch.nn.Embedding(*weights.shape)
    layer.load_state_dict({"weight": torch.tensor(weights)})

    if non_trainable:
        layer.weight.requires_grad = False
    else:
        layer.weight.requires_grad = True

    return layer


class NewRbfKernelBankTF(Layer):
    def __init__(self, mus, sigmas, dim=1, requires_grad=True, **kwargs):
        super().__init__(**kwargs)
        self.size = len(mus)
        self.mus = tf.Variable(mus, trainable=requires_grad, name="mus", dtype=tf.float32)
        self.mus = tf.reshape(self.mus, [1, 1, 1, -1])
        self.sigmas = tf.Variable(sigmas, trainable=requires_grad, name="sigmas", dtype=tf.float32)
        assert dim == 1
        self.permute = [0, 3, 1, 2]

    def count(self):
        return self.size

    def call(self, data, **kwargs):
        # assert len(data.shape) == 3, data.shape
        dtype = data.dtype
        data = tf.cast(data, tf.float32)
        data = tf.expand_dims(data, -1)
        adj = data - self.mus
        out = tf.exp(-0.5 * adj * adj / self.sigmas / self.sigmas)
        out = tf.cast(out, dtype)
        return tf.transpose(out, perm=self.permute)
