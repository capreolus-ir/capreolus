import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.reranker.common import StackedSimilarityMatrix, create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TK_class(nn.Module):
    """
    Adapted from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    """

    def __init__(self, extractor, config):
        super(TK_class, self).__init__()

        self.embeddim = extractor.embeddings.shape[1]
        self.p = config
        self.mus = torch.tensor([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=torch.float)
        self.mu_matrix = self.get_mu_matrix(extractor)
        self.sigma = torch.tensor(0.1, requires_grad=False)

        dropout = 0
        non_trainable = not self.p["finetune"]
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=non_trainable)
        self.cosine_module = StackedSimilarityMatrix(padding=extractor.pad)

        self.position_encoder = PositionalEncoding(self.embeddim)
        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.9, dtype=torch.float32, requires_grad=True))
        encoder_layers = TransformerEncoderLayer(self.embeddim, config["numattheads"], config["ffdim"], dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config["numlayers"])

        self.s_log_fcc = nn.Linear(len(self.mus), 1, bias=False)
        self.s_len_fcc = nn.Linear(len(self.mus), 1, bias=False)
        self.comb_fcc = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.s_log_fcc.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.s_len_fcc.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.comb_fcc.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_mu_matrix(self, extractor):
        """
        Returns a matrix of mu values that can be directly subtracted from the cosine matrix.
        This is the matrix mu in equation 5 in the paper (https://arxiv.org/pdf/2002.01854.pdf)
        """
        qlen = extractor.config["maxqlen"]
        doclen = extractor.config["maxdoclen"]

        mu_matrix = torch.zeros(len(self.mus), qlen, doclen, requires_grad=False)

        for i, mu in enumerate(self.mus):
            mu_matrix[i] = torch.full((qlen, doclen), mu)

        return mu_matrix

    def get_mask(self, embedding):
        """
        Gets a mask of shape (seq_len, seq_len). This is an additive mask, hence masked elements should be -inf
        """
        batch_size = embedding.shape[0]
        seq_len = embedding.shape[1]
        # Get a normal mask of shape (batch_size, seq_len). Entry would be 0 if a seq element should be masked
        mask = ((embedding != torch.zeros(self.embeddim).to(embedding.device)).to(dtype=embedding.dtype).sum(-1) != 0).to(
            dtype=embedding.dtype
        )

        # The square attention mask
        encoder_mask = torch.zeros(batch_size, seq_len, seq_len).to(embedding.device)
        # Set -inf on all rows corresponding to a pad token
        encoder_mask[mask == 0] = float("-inf")
        # Set -inf on all columns corresponding to a pad token (the tricky bit)
        col_mask = mask.reshape(batch_size, 1, seq_len).expand(batch_size, seq_len, seq_len)
        encoder_mask[col_mask == 0] = float("-inf")

        return torch.cat([encoder_mask] * self.p["numattheads"])

    def get_embedding(self, toks):
        """
        Overrides KNRM_Class's get_embedding to return contextualized word embeddings
        """
        embedding = self.embedding(toks)

        # Transformer layers expect input in shape (L, N, E), where L is sequence len, N is batch, E is embed dims
        reshaped_embedding = embedding.permute(1, 0, 2)
        position_encoded_embedding = self.position_encoder(reshaped_embedding)
        # TODO: Mask should be additive
        mask = self.get_mask(embedding) if self.p["usemask"] else None
        contextual_embedding = self.transformer_encoder(position_encoded_embedding, mask).permute(1, 0, 2)
        if self.p["usemixer"]:
            return self.mixer * embedding + (1 - self.mixer) * contextual_embedding
        else:
            return self.p["alpha"] * embedding + (1 - self.p["alpha"]) * contextual_embedding

    def forward(self, doctoks, querytoks, query_idf):
        batches = doctoks.shape[0]
        qlen = querytoks.shape[1]
        doclen = doctoks.shape[1]
        doc = self.get_embedding(doctoks)
        device = doc.device
        query = self.get_embedding(querytoks)
        # cosine_matrix = self.cosine_module.forward(query, doc)
        cosine_matrix = self.cosine_module.forward(query, doc, querytoks, doctoks)
        # cosine_matrix = cosine_matrix.reshape(batches, 1, qlen, doclen)
        cosine_matrix = cosine_matrix.expand(batches, len(self.mus), qlen, doclen)
        kernel_matrix = torch.exp(-torch.pow(cosine_matrix - self.mu_matrix.to(device), 2)) / (2 * torch.pow(self.sigma, 2))
        condensed_kernel_matrix = kernel_matrix.sum(3)
        s_log_k = torch.log2(condensed_kernel_matrix).sum(2)
        s_len_k = condensed_kernel_matrix.sum(2) / doclen

        s_log = self.s_log_fcc(s_log_k)
        s_len = self.s_len_fcc(s_len_k)
        score = self.comb_fcc(torch.cat([s_log, s_len], dim=1))
        return score


@Reranker.register
class TK(Reranker):
    """Sebastian Hofstätter, Markus Zlabinger, and Allan Hanbury. 2019. TU Wien @ TREC Deep Learning '19 -- Simple Contextualization for Re-ranking. In TREC '19."""

    module_name = "TK"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="slowembedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption("gradkernels", True, "backprop through mus and sigmas"),
        ConfigOption("scoretanh", False, "use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)"),
        ConfigOption("singlefc", True, "use single fully connected layer as in paper (True) or 2 fully connected layers (False)"),
        ConfigOption("projdim", 32),
        ConfigOption("ffdim", 100),
        ConfigOption("numlayers", 2),
        ConfigOption("numattheads", 10),
        ConfigOption("alpha", 0.5),
        ConfigOption("usemask", False),
        ConfigOption("usemixer", False),
        ConfigOption("finetune", False, "fine tune the embedding layer"),  # TODO check save when True
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = TK_class(self.extractor, self.config)
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
