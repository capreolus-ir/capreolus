import sys
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from transformers import TFBertForSequenceClassification

from profane import ConfigOption, Dependency
from capreolus.reranker import Reranker


class TFBERTMaxP_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor
        self.bert = TFBertForSequenceClassification.from_pretrained(config["pretrained"], hidden_dropout_prob=0.1)
        self.config = config

    def call(self, x, **kwargs):
        """
        Returns logits of shape [2]
        """
        doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]

        passage_scores = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0]

        return passage_scores

    def predict_step(self, data):
        """
        Scores each passage and applies max pooling over it.
        """
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

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
            passage_scores = tf.math.reduce_sum(tf.nn.softmax(passage_scores), axis=1)
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
    module_name = "TFBERTMaxP"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [
        ConfigOption("pretrained", "bert-base-uncased", "Hugging face transformer pretrained model"),
        ConfigOption("passagelen", 100, "Passage length"),
        ConfigOption("dropout", 0.1, "Dropout for the linear layers in BERT"),
        ConfigOption("stride", 20, "Stride"),
        ConfigOption("aggregation", "max"),
    ]

    def build_model(self):
        self.model = TFBERTMaxP_Class(self.extractor, self.config)
        return self.model
