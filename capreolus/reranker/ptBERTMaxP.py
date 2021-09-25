import random

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class ElectraRelevanceHead(nn.Module):
    """BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.ElectraClassificationHead"""

    def __init__(self, dropout, out_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.out_proj = out_proj

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TFBERTMaxP_Class(nn.Module):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor

        # TODO hidden prob missing below?
        if config["pretrained"] == "electra-base-msmarco":
            self.bert = AutoModelForSequenceClassification.from_pretrained("Capreolus/electra-base-msmarco")
            dropout, fc = self.bert.classifier.dropout, self.bert.classifier.out_proj
            self.bert.classifier = ElectraRelevanceHead(dropout, fc)
        elif config["pretrained"] == "electra-base":
            self.bert = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")
            dropout, fc = self.bert.classifier.dropout, self.bert.classifier.out_proj
            self.bert.classifier = ElectraRelevanceHead(dropout, fc)
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = AutoModelForSequenceClassification.from_pretrained("Capreolus/bert-base-msmarco")
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                config["pretrained"], hidden_dropout_prob=config["hidden_dropout_prob"]
            )

        self.config = config

    def forward(self, doc_input, doc_mask, doc_seg):
        """
        doc_input: (BS, N_PSG, SEQ_LEN) -> [psg-1, psg-2, ..., [PAD], [PAD]]
        """
        batch_size = doc_input.shape[0]
        if "roberta" in self.config["pretrained"]:
            doc_seg = torch.zeros_like(doc_mask)  # since roberta does not have segment input

        if self.training:
            # select one of the passages from extractor.config["num_passages"] for training
            passage_position = (doc_mask * doc_seg).sum(dim=-1)  # (B, P)
            passage_valid_number = (passage_position > 2).sum(dim=-1)  # (B, )
            # explanation about the '2'
            # empty doc_input: [[CLS], Q1, Q2, [SEP], [PAD], [SEP], [PAD], [PAD], [PAD], [PAD]]
            # empty doc_mask:  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
            # empty doc_seg:   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            # empty passage_position: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

            # todo: fix the empty passage issue 
            passage_i = [random.randint(0, max(x - 1, 0)) for x in passage_valid_number] # max(x - 1, 0) as a few passages seems to be empty
            # passage_i = [0 for x in passage_valid_number] # max(x - 1, 0) as a few passages seems to be empty
            # <-- runnable, and takes half time only, but it would decrease MRR@10 for 3 points

            batch_i = torch.arange(batch_size)
            passage_scores = self.bert(
                doc_input[batch_i, passage_i],
                attention_mask=doc_mask[batch_i, passage_i],
                token_type_ids=doc_seg[batch_i, passage_i],
            )[0]
        else:
            passage_scores = self.predict_step(doc_input, doc_mask, doc_seg)

        return passage_scores

    def predict_step(self, doc_input, doc_mask, doc_seg):
        """
        Scores each passage and applies max pooling over it.
        """
        batch_size = doc_input.shape[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        passage_position = (doc_mask * doc_seg).sum(dim=-1)  # (B, P)
        passage_mask = (passage_position > 5).long()  # (B, P)

        doc_input = doc_input.reshape([batch_size * num_passages, maxseqlen])
        doc_mask = doc_mask.reshape([batch_size * num_passages, maxseqlen])
        doc_seg = doc_seg.reshape([batch_size * num_passages, maxseqlen])

        # passage_scores = self.call((doc_input, doc_mask, doc_seg), training=False)[:, 1]
        passage_scores = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 1]
        passage_scores = passage_scores.reshape([batch_size, num_passages])

        if self.config["aggregation"] == "max":
            passage_scores = passage_scores.max(dim=1)[0] # (batch size, )
        # elif self.config["aggregation"] == "first":
        #     passage_scores = passage_scores[:, 0]
        # elif self.config["aggregation"] == "sum":
        #     passage_scores = tf.math.reduce_sum(passage_mask * passage_scores, axis=1)
        # elif self.config["aggregation"] == "avg":
        #     passage_scores = tf.math.reduce_sum(passage_mask * passage_scores, axis=1) / tf.reduce_sum(passage_mask)
        # else:
        #     raise ValueError("Unknown aggregation method: {}".format(self.config["aggregation"]))

        return passage_scores


@Reranker.register
class PTBERTMaxP(Reranker):
    """
    PyTorch implementation of BERT-MaxP.

    Deeper Text Understanding for IR with Contextual Neural Language Modeling. Zhuyun Dai and Jamie Callan. SIGIR 2019.
    https://arxiv.org/pdf/1905.09217.pdf
    """

    module_name = "ptBERTMaxP"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption(
            "pretrained",
            "bert-base-uncased",
            "Pretrained model: bert-base-uncased, bert-base-msmarco, electra-base-msmarco, or HuggingFace supported models",
        ),
        ConfigOption("aggregation", "max"),
        ConfigOption("hidden_dropout_prob", 0.1, "The dropout probability of BERT-like model's hidden layers."),
    ]

    def build_model(self):
        self.model = TFBERTMaxP_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1),
            self.model(d["neg_bert_input"], d["neg_mask"], d["neg_seg"]).view(-1),
        ]

    def test(self, d):
        return self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1)
