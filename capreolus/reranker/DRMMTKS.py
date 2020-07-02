import torch
import torch.nn as nn
import torch.nn.functional as F

from capreolus import ConfigOption
from capreolus.reranker import Reranker
from capreolus.reranker.common import SimilarityMatrix, create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class DRMMTKS_class(nn.Module):
    def __init__(self, extractor, config):
        super(DRMMTKS_class, self).__init__()
        self.topk = config["topk"]
        self.gate_type = config["gateType"]

        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=config["freezeemb"])
        self.simmat = SimilarityMatrix(self.embedding)

        self.ffw = nn.Sequential(nn.Linear(self.topk, 1), nn.Tanh())

        gate_inp_dim = 1 if self.gate_type == "IDF" else self.embedding.weight.size(-1)
        self.gates = nn.Linear(gate_inp_dim, 1, bias=False)  # for idf scalar
        self.output_layer = nn.Linear(1, 1)

        # initialize FC and gate weight in the same way as MatchZoo
        nn.init.uniform_(self.ffw[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.gates.weight, -0.01, 0.01)

    def _term_gate(self, queries, query_idf, q_masks):
        """
        ： param queries: (B, Tq, H)
        :  param query_idf: (B, Tq)
        ： param q_masks: (B, Tq)
        """
        atten_mask = (1 - q_masks) * -1e7

        if self.gate_type == "IDF":
            gate_prob = self.gates(query_idf[:, :, None]).squeeze() + atten_mask  # (B, 1)
        elif self.gate_type == "TV":
            gate_prob = self.gates(queries).squeeze() + atten_mask  # (B, Tq)
        else:
            raise ValueError("Invalid value for histType: histType should be 'CH', 'NH', or 'LCH'")

        gate_prob = F.softmax(gate_prob, dim=1)  # (B, Tq)
        return gate_prob

    def forward(self, doc, query, query_idf):
        query_sent_mask = (query != 0).float()  # (B, Q)
        sent_pad_pos = (doc == 0).float()  # (B, D)
        query_idf = query_idf.float()

        cos_mat = self.simmat(query, doc)
        topk, _ = torch.topk(cos_mat, k=self.topk, dim=-1)  # (B, Q, k)
        ffw_vec = self.ffw(topk).squeeze()  # (B, Q)

        w = self._term_gate(query, query_idf, query_sent_mask)  # （B, Q）
        doc = (w * ffw_vec).sum(dim=-1, keepdim=True)  # (B, 1)
        score = self.output_layer(doc)

        return score


dtype = torch.FloatTensor


@Reranker.register
class DRMMTKS(Reranker):
    """Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. 2016. A Deep Relevance Matching Model for Ad-hoc Retrieval. In CIKM'16."""

    # reference: https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/drmmtks.py
    module_name = "DRMMTKS"

    config_spec = [
        ConfigOption("topk", 10, "number of bins in matching histogram"),
        ConfigOption("gateType", "IDF", "term gate type: TV or IDF"),
        ConfigOption("freezeemb", True, "term gate type: TV or IDF"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = DRMMTKS_class(self.extractor, self.config)

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
