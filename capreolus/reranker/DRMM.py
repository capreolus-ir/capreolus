import torch
import torch.nn as nn
import torch.nn.functional as F

from capreolus import ConfigOption
from capreolus.reranker import Reranker
from capreolus.reranker.common import SimilarityMatrix, create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class DRMM_class(nn.Module):
    def __init__(self, extractor, config):
        super(DRMM_class, self).__init__()
        # self.p = config
        self.nbins = config["nbins"]
        self.nodes = config["nodes"]
        self.hist_type = config["histType"]
        self.gate_type = config["gateType"]

        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        self.simmat = SimilarityMatrix(self.embedding)

        self.ffw = nn.Sequential(nn.Linear(self.nbins + 1, self.nodes), nn.Tanh(), nn.Linear(self.nodes, 1), nn.Tanh())

        emb_dim = self.embedding.weight.size(-1)
        if self.gate_type == "IDF":
            self.gates = nn.Linear(1, 1, bias=False)  # for idf scalar
        elif self.gate_type == "TV":
            self.gates = nn.Linear(emb_dim, 1, bias=False)
        else:
            raise ValueError("Invalid value for gateType: gateType should be either IDF or TV")
        self.output_layer = nn.Linear(1, 1)

        # initialize FC and gate weight in the same way as MatchZoo
        nn.init.uniform_(self.ffw[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.ffw[2].weight, -0.1, 0.1)
        nn.init.uniform_(self.gates.weight, -0.01, 0.01)

    def _hist_map(self, queries, documents, d_masks):
        """
        Args:
            queries: (B, Tq, H)
            documents: (B, Td, H)
            d_masks: (B, Td)

        Returns: (B, Tq, 6)
        """

        # compute cos similarity
        # q_norm = torch.sqrt((queries * queries).sum(dim=-1) + 1e-7)[:, :, None] + 1e-7
        # d_norm = torch.sqrt((documents * documents).sum(dim=-1) + 1e-7)[:, None, :] + 1e-7
        # sim_matrix = torch.bmm(queries, documents.transpose(2, 1))  # (B, Tq, Td)
        # sim_matrix = sim_matrix / q_norm
        # sim_matrix = sim_matrix / d_norm  # (B, Tq, Td)

        sim_matrix = self.simmat(queries, documents)
        sim_matrix += (1 - d_masks[:, None, :]) * 1e7  # assign large number on <PAD> pos
        hist = torch.zeros([sim_matrix.size(0), sim_matrix.size(1), self.nbins + 1], dtype=torch.float)

        idxs = list(range(self.nbins))
        bin_upperbounds = torch.linspace(-1, 1, self.nbins + 1)[1:].to(queries.device)
        for i, bin_upperbound in zip(idxs, bin_upperbounds):
            hist[:, :, i] = (sim_matrix < bin_upperbound).sum(dim=-1)
        hist[:, :, -1] = ((sim_matrix > 0.999) * (sim_matrix < 1.001)).sum(dim=-1)

        for i in list(range(self.nbins - 1, 0, -1)):  # exclude idx=config['nbins'] and idx=0
            hist[:, :, i] -= hist[:, :, i - 1]

        hist += 1

        if self.hist_type == "NH":
            hist_sum = hist.sum(dim=-1)  # (B, T)
            hist = hist / hist_sum[:, :, None]
        elif self.hist_type == "LCH":
            hist = torch.log(hist)
        elif self.hist_type != "CH":
            raise ValueError("Invalid value for gateType: gateType should be either IDF or TV")

        return hist

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

    def forward(self, sentence, query_sentence, query_idf):
        query_sent_mask = (query_sentence != 0).to(sentence.device).float()  # (B, Tq)
        sent_mask = (sentence != 0).to(sentence.device).float()  # (B, Td)

        hist_vec = self._hist_map(query_sentence, sentence, sent_mask).to(sentence.device)
        ffw_vec = self.ffw(hist_vec).squeeze().to(sentence.device)  # (B, T1)

        query_idf = query_idf.float()
        query_x = self.embedding(query_sentence).to(sentence.device).float()
        w = self._term_gate(query_x, query_idf, query_sent_mask)  # （B, T1）

        x = (w * ffw_vec).sum(dim=-1, keepdim=True)  # (B, 1)

        score = self.output_layer(x)

        return score


dtype = torch.FloatTensor


@Reranker.register
class DRMM(Reranker):
    """Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. 2016. A Deep Relevance Matching Model for Ad-hoc Retrieval. In CIKM'16."""

    module_name = "DRMM"

    config_spec = [
        ConfigOption("nbins", 29, "number of bins in matching histogram"),
        ConfigOption("nodes", 5, "hidden layer dimension for feed forward matching network"),
        ConfigOption("histType", "LCH", "histogram type: CH, NH, or LCH"),
        ConfigOption("gateType", "IDF", "term gate type: TV or IDF"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = DRMM_class(self.extractor, self.config)

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
