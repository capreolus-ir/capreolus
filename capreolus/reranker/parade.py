import sys
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers.modeling_tf_bert import TFBertLayer

from profane import ConfigOption, Dependency
from capreolus.reranker import Reranker


class TFParade_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFParade_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.transformer_layer_1 = TFBertLayer(self.bert.config)
        self.transformer_layer_2 = TFBertLayer(self.bert.config)
        # self.num_passages = (self.extractor.cfg["maxdoclen"] - config["passagelen"]) // self.config["stride"]
        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.bert.config.hidden_size,))
        if config["aggregation"] == "maxp":
            self.aggregation = self.aggregate_using_maxp
        elif config["aggregation"] == "transformer":
            self.aggregation = self.aggregate_using_transformer
            input_embeddings = self.bert.get_input_embeddings()
            self.initial_cls_embedding = input_embeddings([101, None, None, None])

    def aggregate_using_maxp(self, cls):
        """
        cls has the shape [B, num_passages, hidden_size]
        """
        expanded_cls = tf.reshape(cls, [-1, self.num_passages, self.bert.config.hidden_size])
        batch_size = tf.shape(expanded_cls)[0]
        aggregated = tf.reduce_max(expanded_cls, axis=1)

        return aggregated

    def aggregate_using_transformer(self, cls):
        expanded_cls = tf.reshape(cls, [-1, self.num_passages, self.bert.config.hidden_size])
        batch_size = tf.shape(expanded_cls)[0]
        tiled_initial_cls = tf.tile(self.initial_cls_embedding, multiples=[batch_size, 1])
        # TODO: Check with Canjia if the concat order is correct
        merged_cls = tf.concat((expanded_cls, tf.expand_dims(tiled_initial_cls, axis=1)), axis=1)
        tf.debugging.assert_equal(tf.shape(merged_cls), [batch_size, self.num_passages + 1, self.bert.config.hidden_size])

        full_position_embeddings = tf.compat.v1.get_variable(
            name="passage_position_embedding",
            shape=[self.num_passages + 1, self.bert.config.hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
        )
        full_position_embeddings = tf.expand_dims(full_position_embeddings, axis=0)
        print("full_position_embeddings shape is {}".format(tf.shape(full_position_embeddings)))
        merged_cls += full_position_embeddings
        print("merged_cls shape is now: {}".format(tf.shape(merged_cls)))

        attention_mask = tf.sequence_mask(batch_size, self.num_passages + 1, dtype=tf.float32)
        attention_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), [1, self.num_passages + 1, 1])

        (transformer_out_1,) = self.transformer_layer_1((merged_cls, attention_mask, None, None))
        (transformer_out_2,) = self.transformer_layer_2((transformer_out_1, attention_mask, None, None))
        print("transformer_out_2 shape is {}".format(tf.shape(transformer_out_2)))

        aggregated = transformer_out_2[:, 0, :]
        print("aggregated shape is {}".format(tf.shape(aggregated)))
        return aggregated

    def call(self, x, **kwargs):
        doc_input, doc_mask, doc_seg = x[0], x[1], x[2]
        batch_size = tf.shape(doc_input)[0]

        doc_input = tf.reshape(doc_input, [batch_size * self.num_passages, self.maxseqlen])
        doc_mask = tf.reshape(doc_mask, [batch_size * self.num_passages, self.maxseqlen])
        doc_seg = tf.reshape(doc_seg, [batch_size * self.num_passages, self.maxseqlen])

        cls = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 0, :]
        aggregated = self.aggregation(cls)
        # tf.debugging.assert_equal(tf.shape(cls), [batch_size * self.num_passages, self.bert.config.hidden_size])
        # (transformer_out_1, ) = self.transformer_layer_1((cls, None, None, None))
        # print("transformer_out_2 has the shape {}".format(tf.shape(transformer_out_1)))
        # (transformer_out_2, ) = self.transformer_layer_2((transformer_out_1, None, None, None))
        # print("transformer_out_2 shape is {}".format(tf.shape(transformer_out_2)))
        # transformer_out_2 = tf.reshape(transformer_out_2, [batch_size, self.num_passages, self.bert.config.hidden_size])
        # scores = self.linear(transformer_out_2)

        return self.linear(aggregated)

    def predict_step(self, data):
        """
        Scores each passage and applies max pooling over it.
        """
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

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
        ConfigOption("aggregation", "maxp"),
    ]

    def build_model(self):
        self.model = TFParade_Class(self.extractor, self.config)
        return self.model
