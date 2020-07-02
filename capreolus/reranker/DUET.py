import torch
import torch.nn as nn

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.reranker.common import create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class LocalModel(nn.Module):
    def __init__(self, p):
        super(LocalModel, self).__init__()
        self.p = p
        if p["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif p["activation"] == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unexpected activation: should be either tanh or relu")

        q_len, doc_len = p["extractor"]["maxqlen"], p["extractor"]["maxdoclen"]
        dropoutrate = p["dropoutrate"]
        self.conv = nn.Sequential(nn.Conv2d(1, p["nfilters"], (1, doc_len)), self.activation)  # (B, 1, Q, D) -> (B, H, Q, 1)

        self.ffw = nn.Sequential(
            nn.Linear(q_len * p["nfilters"], p["lmhidden"]), self.activation, nn.Dropout(dropoutrate), nn.Linear(p["lmhidden"], 1)
        )

    def exact_match(self, m1, m2):
        """
        m1: (B, len1)
        m2: (B, len2)
        """
        len1, len2 = m1.size(1), m2.size(1)
        m1_expand = torch.stack([m1] * len2, dim=2)
        m2_expand = torch.stack([m2] * len1, dim=1)
        return (m1_expand == m2_expand).float()

    def forward(self, documents, queries, query_idf):
        """
        queries: (B, nq)
        documents: (B, nd)
        query_idf: (B, nq)
        """
        lm_matrix = self.exact_match(queries, documents)  # (B, nq, nd)
        if self.p["idfweight"]:
            lm_matrix = lm_matrix * query_idf[:, :, None]

        lm_x = self.conv(lm_matrix.unsqueeze(1)).squeeze()  # (B, H1, nq)
        lm_score = self.ffw(lm_x.view(lm_x.size(0), -1))
        return lm_score


class DistributedModel(nn.Module):
    def __init__(self, extractor, p):
        super(DistributedModel, self).__init__()
        if p["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif p["activation"] == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unexpected activation: should be either tanh or relu")

        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        embsize = self.embedding.weight.shape[-1]
        dropoutrate = p["dropoutrate"]

        self.q_conv = nn.Sequential(
            nn.Conv2d(1, p["nfilters"], (3, embsize)),
            self.activation,
            nn.Dropout(dropoutrate),
            nn.MaxPool2d((2, 1), stride=(1, 1)),
        )

        self.q_ffw = nn.Sequential(nn.Linear(p["nfilters"], p["nfilters"]))

        # (B, 1, Q, V) -> (B, H, Q', 1)
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(1, p["nfilters"], (3, embsize)),
            self.activation,
            nn.Dropout(dropoutrate),
            nn.MaxPool2d((100, 1), stride=(1, 1)),
        )

        self.d_conv2 = nn.Sequential(
            nn.Conv2d(1, p["nfilters"], (p["nfilters"], 1)),  # (B, 1, H, Q') -> (B, H, 1, Q')
            self.activation,
            nn.Dropout(dropoutrate),
        )

        self.ffw_1 = nn.Sequential(nn.Linear(p["nhidden"], 1), self.activation)
        self.ffw_2 = nn.Sequential(nn.Dropout(dropoutrate), nn.Linear(p["nfilters"], 1))

    def forward(self, documents, queries):
        # dm query
        dm_q = self.embedding(queries).unsqueeze(1)  # (B, 1, nq, D)
        dm_q = self.q_conv(dm_q).squeeze()  # (B, H)
        dm_q = self.q_ffw(dm_q)  # (B, H)

        # dm document
        dm_d = self.embedding(documents).unsqueeze(1)  # (B, 1, nd, D)
        dm_d = self.d_conv1(dm_d).squeeze()  # (B, H, 699)
        dm_d = self.d_conv2(dm_d.unsqueeze(1)).squeeze()  # (B, H, 699) -> (B, 1, H, 699) -> (B, H, 1, 699) -> (B, H, 699)

        # aggregate dm_q & dm_d
        dm_x = dm_q.unsqueeze(2) * dm_d  # (B, H, 1) * (B, H, 699)
        dm_x = self.ffw_1(dm_x).squeeze()  # -> (B, H, 1) -> (B, H)
        dm_score = self.ffw_2(dm_x)  # (B, H) -> (B, H) -> (B, 1)

        return dm_score


class DUET_class(nn.Module):
    def __init__(self, extractor, p):
        super(DUET_class, self).__init__()
        self.lm = LocalModel(p)
        self.dm = DistributedModel(extractor, p)

    def forward(self, documents, queries, query_idf):
        """
        queries: (B, nq)
        documents: (B, nd)
        """
        lm_score = self.lm(documents, queries, query_idf)
        dm_score = self.dm(documents, queries)

        return lm_score + dm_score


@Reranker.register
class DUET(Reranker):
    """Bhaskar Mitra, Fernando Diaz, and Nick Craswell. 2017. Learning to Match using Local and Distributed Representations of Text for Web Search. In WWW'17."""

    module_name = "DUET"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="slowembedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption("nfilters", 10, "number of filters for both local and distrbuted model"),
        ConfigOption("lmhidden", 30, "ffw hidden layer dimension for local model"),
        ConfigOption("nhidden", 699, "ffw hidden layer dimension for local model"),
        ConfigOption("idfweight", True, "whether to weight each query word with its idf value in local model"),
        ConfigOption("dropoutrate", 0.5, "dropout probability"),
        ConfigOption("activation", "relu", "ffw layer activation: tanh or relu"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = DUET_class(self.extractor, self.config)
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
