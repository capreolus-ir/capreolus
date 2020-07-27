import sys
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from transformers import TFBertModel
from transformers.modeling_tf_bert import TFBertLayer

from profane import ConfigOption, Dependency
from capreolus.reranker import Reranker


class TFParade_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFParade_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config
        self.bert = TFBertModel.from_pretrained(config["pretrained"], hidden_dropout_prob=0.1)
        self.transformer_layer_1 = TFBertLayer(self.bert.config)
        self.transformer_layer_2 = TFBertLayer(self.bert.config)
        # self.num_passages = (self.extractor.cfg["maxdoclen"] - config["passagelen"]) // self.config["stride"]
        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.bert.config.hidden_size,))

    def call(self, x, **kwargs):
        doc_input, doc_mask, doc_seg = x[0], x[1], x[2]
        batch_size = tf.shape(doc_input)[0]

        doc_input = tf.reshape(doc_input, [batch_size * self.num_passages, self.maxseqlen])
        doc_mask = tf.reshape(doc_mask, [batch_size * self.num_passages, self.maxseqlen])
        doc_seg = tf.reshape(doc_seg, [batch_size * self.num_passages, self.maxseqlen])

        cls = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 0]
        tf.debugging.assert_equal(tf.shape(cls), (batch_size, self.num_passages * self.bert.config.hidden_size))
        cls = tf.reshape(cls, [batch_size, self.num_passages, self.bert.config.hidden_size])

        (transformer_out1,) = self.transformer_layer_1((cls, None, None))
        (transformer_out2,) = self.transformer_layer_2((transformer_out1, None, None))
        final_cls = tf.reshape(transformer_out2[:, 0], [batch_size, self.bert.config.hidden_size])

        score = tf.reshape(self.linear(final_cls), [batch_size, 1])

        return score

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

        doc_scores = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), training=False)

        return doc_scores

    def score(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        return self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)

    def score_pair(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        pos_score = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)
        neg_score = self.call((negdoc_bert_input, negdoc_mask, negdoc_seg), **kwargs)

        return pos_score, neg_score


@Reranker.register
class TFParade(Reranker):
    module_name = "parade"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="pooledbertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [
        ConfigOption("pretrained", "bert-base-uncased", "Hugging face transformer pretrained model"),
        ConfigOption("passagelen", 100, "Passage length"),
        ConfigOption("dropout", 0.1, "Dropout for the linear layers in BERT"),
        ConfigOption("stride", 20, "Stride"),
    ]

    def build_model(self):
        self.model = TFParade_Class(self.extractor, self.config)
        return self.model
