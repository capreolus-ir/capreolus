import math

import torch
from torch import nn
from transformers import BertModel, ElectraModel, AutoModel

from capreolus import ConfigOption, Dependency, get_logger
from capreolus.reranker import Reranker
from capreolus.reranker.common import RbfKernelBank

logger = get_logger(__name__)


class CEDRKNRM_Class(nn.Module):
    def __init__(self, extractor, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config

        if config["pretrained"] == "electra-base-msmarco":
            self.bert = ElectraModel.from_pretrained(
                "Capreolus/electra-base-msmarco", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "electra-base":
            self.bert = ElectraModel.from_pretrained(
                "google/electra-base-discriminator", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = BertModel.from_pretrained(
                "Capreolus/bert-base-msmarco", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "bert-base-uncased":
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        else:
            self.bert = AutoModel.from_pretrained(
                config["pretrained"], hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )

        self.hidden_size = self.bert.config.hidden_size
        mus = list(self.config["mus"]) + [1.0]
        sigmas = [self.config["sigma"] for _ in self.config["mus"]] + [0.01]
        logger.debug("mus: %s", mus)
        self.kernels = RbfKernelBank(mus, sigmas, dim=1, requires_grad=self.config["gradkernels"])

        if -1 in self.config["simmat_layers"]:
            assert len(self.config["simmat_layers"]) == 1
            assert self.config["cls"] is not None
            self._compute_simmat = False
            combine_size = 0
        else:
            self._compute_simmat = True
            combine_size = self.kernels.count() * len(self.config["simmat_layers"])

        assert self.config["cls"] in ("avg", "max", None)
        if self.config["cls"]:
            combine_size += self.hidden_size

        # use weight init from PyTorch 0.4
        if config["combine_hidden"] == 0:
            combine_steps = [nn.Linear(combine_size, 1)]
            stdv = 1.0 / math.sqrt(combine_steps[0].weight.size(1))
            combine_steps[0].weight.data.uniform_(-stdv, stdv)
        else:
            combine_steps = [nn.Linear(combine_size, config["combine_hidden"]), nn.Linear(config["combine_hidden"], 1)]
            stdv = 1.0 / math.sqrt(combine_steps[0].weight.size(1))
            combine_steps[0].weight.data.uniform_(-stdv, stdv)
            stdv = 1.0 / math.sqrt(combine_steps[-1].weight.size(1))
            combine_steps[-1].weight.data.uniform_(-stdv, stdv)

        self.combine = nn.Sequential(*combine_steps)

        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        # TODO we include SEP in maxqlen due to the way the simmat is constructed... (and another SEP in document)
        # (maxqlen is the actual query length and does not count CLS or SEP)
        self.maxqlen = extractor.config["maxqlen"] + 1
        # decreased by 1 because we remove CLS before generating embeddings
        self.maxdoclen = self.maxseqlen - 1
        self.one = nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _cos_simmat(self, a, b, amask, bmask):
        # based on cos_simmat from https://github.com/Georgetown-IR-Lab/OpenNIR/blob/master/onir/modules/interaction_matrix.py
        # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
        BAT, A, B = a.shape[0], a.shape[1], b.shape[1]
        a_denom = a.norm(p=2, dim=2).reshape(BAT, A, 1) + 1e-9  # avoid 0div
        b_denom = b.norm(p=2, dim=2).reshape(BAT, 1, B) + 1e-9  # avoid 0div
        result = a.bmm(b.permute(0, 2, 1)) / (a_denom * b_denom)
        result = result * amask.reshape(BAT, A, 1)
        result = result * bmask.reshape(BAT, 1, B)
        return result

    def masked_simmats(self, embeddings, bert_mask, bert_segments):
        # segment 0 contains '[CLS] query [SEP]' and segment 1 contains 'document [SEP]'
        query_mask = bert_mask * torch.where(bert_segments == 0, self.one, self.zero)
        padded_query = (query_mask.unsqueeze(2) * embeddings)[:, : self.maxqlen]
        query_mask = query_mask[:, : self.maxqlen]

        doc_mask = bert_mask * torch.where(bert_segments == 1, self.one, self.zero)
        # padded_doc length is maxsdoclen; zero padding on both left and right of doc
        padded_doc = doc_mask.unsqueeze(2) * embeddings

        # (maxqlen, maxdoclen)
        simmat = self._cos_simmat(padded_query, padded_doc, query_mask, doc_mask)
        return simmat, doc_mask, query_mask

    def knrm(self, bert_output, bert_mask, bert_segments, batch_size):
        # create similarity matrix for each passage (skipping CLS)
        passage_simmats, passage_doc_mask, passage_query_mask = self.masked_simmats(
            bert_output[:, 1:], bert_mask[:, 1:], bert_segments[:, 1:]
        )
        passage_simmats = passage_simmats.view(batch_size, self.num_passages, self.maxqlen, self.maxdoclen)
        passage_doc_mask = passage_doc_mask.view(batch_size, self.num_passages, 1, -1)

        # concat similarity matrices along document dimension; query mask is the same across passages
        doc_simmat = torch.cat([passage_simmats[:, PIDX, :, :] for PIDX in range(self.num_passages)], dim=2)
        doc_mask = torch.cat([passage_doc_mask[:, PIDX, :, :] for PIDX in range(self.num_passages)], dim=2)
        query_mask = passage_query_mask.view(batch_size, self.num_passages, -1, 1)[:, 0, :, :]

        # KNRM on similarity matrix
        prepooled_doc = self.kernels(doc_simmat)
        prepooled_doc = prepooled_doc * doc_mask.view(batch_size, 1, 1, -1) * query_mask.view(batch_size, 1, -1, 1)

        # sum over document
        knrm_features = prepooled_doc.sum(dim=3)
        knrm_features = torch.log(torch.clamp(knrm_features, min=1e-10)) * 0.01

        # sum over query
        knrm_features = knrm_features.sum(dim=2)

        return knrm_features

    def diffir_weights(self, bert_input, bert_mask, bert_segments):
        batch_size = bert_input.shape[0]
        bert_input = bert_input.view((batch_size * self.num_passages, self.maxseqlen))
        bert_mask = bert_mask.view((batch_size * self.num_passages, self.maxseqlen))
        bert_segments = bert_segments.view((batch_size * self.num_passages, self.maxseqlen))

        # get BERT embeddings (including CLS) for each passage
        # TODO switch to hgf's ModelOutput after bumping tranformers version
        outputs = self.bert(bert_input, attention_mask=bert_mask, token_type_ids=bert_segments)
        if self.config["pretrained"].startswith("bert-"):
            outputs = (outputs[0], outputs[2])
        bert_output, all_layer_output = outputs

        # Create the simmat from the final layer
        assert self.config["simmat_layers"] == 12, "simmat laters: {}".format(self.config["simmat_layers"])
        passage_simmats, passage_doc_mask, passage_query_mask = self.masked_simmats(
            all_layer_output[self.config["simmat_layers"]][:, 1:], bert_mask[:, 1:], bert_segments[:, 1:]
        )

        assert passage_simmats.shape == (self.num_passages, self.maxqlen, self.maxdoclen), "shape: {}".format(passage_simmats.shape)

        passage_doc_mask = passage_doc_mask.view(batch_size, self.num_passages, 1, -1)

        return passage_simmats, passage_doc_mask

    def forward(self, bert_input, bert_mask, bert_segments):
        batch_size = bert_input.shape[0]
        bert_input = bert_input.view((batch_size * self.num_passages, self.maxseqlen))
        bert_mask = bert_mask.view((batch_size * self.num_passages, self.maxseqlen))
        bert_segments = bert_segments.view((batch_size * self.num_passages, self.maxseqlen))

        # get BERT embeddings (including CLS) for each passage
        # TODO switch to hgf's ModelOutput after bumping tranformers version
        outputs = self.bert(bert_input, attention_mask=bert_mask, token_type_ids=bert_segments)
        if self.config["pretrained"].startswith("bert-"):
            outputs = (outputs[0], outputs[2])
        bert_output, all_layer_output = outputs

        # average CLS embeddings to create the CLS feature
        cls = bert_output[:, 0, :]

        if self.config["cls"] == "max":
            cls_features = cls.view(batch_size, self.num_passages, self.hidden_size).max(dim=1)[0]
        elif self.config["cls"] == "avg":
            cls_features = cls.view(batch_size, self.num_passages, self.hidden_size).mean(dim=1)

        # create KNRM features for each output layer
        if self._compute_simmat:
            layer_knrm_features = [
                self.knrm(all_layer_output[LIDX], bert_mask, bert_segments, batch_size) for LIDX in self.config["simmat_layers"]
            ]

        # concat CLS+KNRM features and pass to linear layer
        if self.config["cls"] and self._compute_simmat:
            all_features = torch.cat([cls_features] + layer_knrm_features, dim=1)
        elif self._compute_simmat:
            all_features = torch.cat(layer_knrm_features, dim=1)
        elif self.config["cls"]:
            all_features = cls_features
        else:
            raise ValueError("invalid config: %s" % self.config)

        score = self.combine(all_features)
        return score


@Reranker.register
class CEDRKNRM(Reranker):
    """
    PyTorch implementation of CEDR-KNRM.
    Equivalant to BERT-KNRM when cls=None.

    CEDR: Contextualized Embeddings for Document Ranking
    Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. SIGIR 2019.
    https://arxiv.org/pdf/1904.07094
    """

    module_name = "CEDRKNRM"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="pooledbertpassage"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption(
            "pretrained",
            "electra-base",
            "Pretrained model: bert-base-uncased, bert-base-msmarco, electra-base, or electra-base-msmarco",
        ),
        ConfigOption("mus", [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9], "mus", value_type="floatlist"),
        ConfigOption("sigma", 0.1, "sigma"),
        ConfigOption("gradkernels", True, "tune mus and sigmas"),
        ConfigOption("hidden_dropout_prob", 0.1, "The dropout probability of BERT-like model's hidden layers."),
        ConfigOption("simmat_layers", "0..12,1", "Layer outputs to include in similarity matrix", value_type="intlist"),
        ConfigOption("combine_hidden", 1024, "Hidden size to use with combination FC layer (0 to disable)"),
        ConfigOption("cls", "avg", "Handling of CLS token: avg, max, or None"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = CEDRKNRM_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1),
            self.model(d["neg_bert_input"], d["neg_mask"], d["neg_seg"]).view(-1),
        ]

    def test(self, d):
        return self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1)

    def diffir_weights(self, d):
        return self.model.diffir_weights(d["pos_bert_input"], d["pos_mask"], d["pos_seg"])