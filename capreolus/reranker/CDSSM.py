import torch
from torch import nn

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.reranker.common import create_emb_layer


class CDSSM_class(nn.Module):
    def __init__(self, extractor, config):
        super(CDSSM_class, self).__init__()
        self.p = config

        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        self.conv = nn.Sequential(
            nn.Conv1d(1, config["nfilter"], config["nkernel"]), nn.ReLU(), nn.Dropout(config["dropoutrate"])
        )
        self.ffw = nn.Linear(config["nkernel"], config["nhiddens"])

        self.output_layer = nn.Sequential(nn.Sigmoid())

    def forward(self, sentence, query):
        # query = query.to(device)                # (B, Q)
        # sentence = sentence.to(device)          # (B, D)
        device = sentence.device
        query = self.embedding(query)  # (B, Q)
        sentence = self.embedding(sentence)  # (B, D)

        # pad sentence so its timestep is divisible by W
        B, Q, H = query.size()
        W = self.p["windowsize"]

        # pad query
        npad = W - query.size(1) % W
        pads = torch.zeros(B, npad, H).to(device)
        # assert query.size(1) % W == 0
        # assert sentence.size(1) % W == 0
        query = torch.cat([query, pads], dim=1)  # (B, K_q*W, H)

        # pad document
        npad = W - sentence.size(1) % W
        pads = torch.zeros(B, npad, H).to(device)
        sentence = torch.cat([sentence, pads], dim=1)  # (B, K_d*W, H)

        query = query.reshape(B, -1, W * H)  # (B, K_q, Q*H)
        sentence = sentence.reshape(B, -1, W * H)  # (B, K_d, Q*H)

        # major part of CDSSM
        query = torch.cat([self.conv(query[:, i : i + 1, :]) for i in range(query.size(1))], dim=1)  # (B, K_q, Q*H/Kernel)

        sentence = torch.cat(  # (B, K_d, Q*H/Kernel)
            [self.conv(sentence[:, i : i + 1, :]) for i in range(sentence.size(1))], dim=1
        )

        # 'max pooling'
        query, _ = torch.max(query, dim=1)  # (B, Q*H/Kernel)
        sentence, _ = torch.max(sentence, dim=1)  # (B, Q*H/Kernel)

        query_norm = query.norm(dim=-1)[:, None] + 1e-7
        sentence_norm = sentence.norm(dim=-1)[:, None] + 1e-7

        query = query / query_norm
        sentence = sentence / sentence_norm

        cos_x = (query * sentence).sum(dim=-1, keepdim=True)

        score = self.output_layer(cos_x)

        return score


dtype = torch.FloatTensor


@Reranker.register
class CDSSM(Reranker):
    """Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, and Grégoire Mesnil. 2014. A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. In CIKM'14."""

    module_name = "CDSSM"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="slowembedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption("nkernel", 3, "kernel dimension in conv"),
        ConfigOption("nfilter", 1, "number of filters in conv"),
        ConfigOption("nhiddens", 30, "hidden layer dimension for ffw layer"),
        ConfigOption("windowsize", 3, "number of query/document words to concatenate before conv"),
        ConfigOption("dropoutrate", 0, "dropout rate for conv"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = CDSSM_class(self.extractor, self.config)

        return self.model

    def score(self, d):
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [self.model(pos_sentence, query_sentence).view(-1), self.model(neg_sentence, query_sentence).view(-1)]

    def test(self, data):
        query_sentence, pos_sentence = data["query"], data["posdoc"]

        return self.model(pos_sentence, query_sentence).view(-1)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)
