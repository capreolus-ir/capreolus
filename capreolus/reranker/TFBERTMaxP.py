import tensorflow as tf
import numpy as np
from transformers import TFAutoModelForSequenceClassification

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker


class TFElectraRelevanceHead(tf.keras.layers.Layer):
    """ BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.TFElectraClassificationHead """

    def __init__(self, dropout, out_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.out_proj = out_proj

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TFBERTMaxP_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor

        # TODO hidden prob missing below?
        if config["pretrained"] == "electra-base-msmarco":
            self.bert = TFAutoModelForSequenceClassification.from_pretrained("Capreolus/electra-base-msmarco")
            dropout, fc = self.bert.classifier.dropout, self.bert.classifier.out_proj
            self.bert.classifier = TFElectraRelevanceHead(dropout, fc)
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = TFAutoModelForSequenceClassification.from_pretrained("Capreolus/bert-base-msmarco")
        else:
            self.bert = TFAutoModelForSequenceClassification.from_pretrained(
                config["pretrained"], hidden_dropout_prob=config["hidden_dropout_prob"]
            )

        self.config = config

    def call(self, x, **kwargs):
        """
        Returns logits of shape [2]
        """
        doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]
        if "roberta" in self.config["pretrained"]:
            doc_seg = tf.zeros_like(doc_mask)  # since roberta does not have segment input
        passage_scores = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0]

        return passage_scores

    def extract_weights(self, data):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        passage_position = tf.reduce_sum(posdoc_mask * posdoc_seg, axis=-1)  # (B, P)
        passage_mask = tf.cast(tf.greater(passage_position, 5), tf.float32)  # (B, P)

        posdoc_bert_input = tf.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        passage_scores = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), training=False)[:, 1]
        passage_scores = tf.reshape(passage_scores, [batch_size, num_passages])

        return passage_scores

    def predict_step(self, data):
        """
        Scores each passage and applies max pooling over it.
        """
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        passage_position = tf.reduce_sum(posdoc_mask * posdoc_seg, axis=-1)  # (B, P)
        passage_mask = tf.cast(tf.greater(passage_position, 5), tf.float32)  # (B, P)

        posdoc_bert_input = tf.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        passage_scores = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), training=False)[:, 1]
        passage_scores = tf.reshape(passage_scores, [batch_size, num_passages])

        if self.config["aggregation"] == "max":
            passage_scores = tf.math.reduce_max(passage_scores, axis=1)
        elif self.config["aggregation"] == "first":
            passage_scores = passage_scores[:, 0]
        elif self.config["aggregation"] == "sum":
            passage_scores = tf.math.reduce_sum(passage_mask * passage_scores, axis=1)
        elif self.config["aggregation"] == "avg":
            passage_scores = tf.math.reduce_sum(passage_mask * passage_scores, axis=1) / tf.reduce_sum(passage_mask)
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.config["aggregation"]))

        return passage_scores

    def score(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x
        return self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)

    def score_pair(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        pos_score = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)[:, 1]
        neg_score = self.call((negdoc_bert_input, negdoc_mask, negdoc_seg), **kwargs)[:, 1]

        return pos_score, neg_score


@Reranker.register
class TFBERTMaxP(Reranker):
    """
    TensorFlow implementation of BERT-MaxP.

    Deeper Text Understanding for IR with Contextual Neural Language Modeling. Zhuyun Dai and Jamie Callan. SIGIR 2019.
    https://arxiv.org/pdf/1905.09217.pdf
    """

    module_name = "TFBERTMaxP"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
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

    def weights_to_weighted_char_ranges(self, docid, passage_scores):
        assert passage_scores.shape == (self.extractor.config["numpassages"], ), "passage scores shape is {}".format(passage_scores.shape)
        diffir_weights = []
        doc_offsets = self.extractor.docid_to_doc_offsets_obj[docid]

        max_weight = -np.inf
        for passage_id in range(self.extractor.config["numpassages"]):
            if passage_id not in self.extractor.docid_to_passage_begin_token_obj[docid]:
                continue

            passage_begin_token_id = self.extractor.docid_to_passage_begin_token_obj[docid][passage_id]
            passage_begin = doc_offsets[passage_begin_token_id][0]
            if passage_id + 1 in self.extractor.docid_to_passage_begin_token_obj[docid]:
                next_passage_begin_token_id = self.extractor.docid_to_passage_begin_token_obj[docid][passage_id + 1]
                next_passage_begin = doc_offsets[next_passage_begin_token_id]
                passage_end = next_passage_begin[0]
            else:
                passage_end = doc_offsets[-1][1]

            curr_weight = passage_scores[passage_id].numpy().item()
            if curr_weight > max_weight:
                max_weight = curr_weight
                diffir_weights = [[passage_begin, passage_end, passage_scores[passage_id].numpy().item()]]

        return diffir_weights
