import tensorflow as tf
from transformers import TFBertModel, TFElectraModel
from transformers.modeling_tf_bert import TFBertLayer

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker


class TFParade_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFParade_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config

        if config["pretrained"] == "electra-base-msmarco":
            self.bert = TFElectraModel.from_pretrained("Capreolus/electra-base-msmarco")
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = TFBertModel.from_pretrained("Capreolus/bert-base-msmarco")
        elif config["pretrained"] == "bert-base-uncased":
            self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        else:
            raise ValueError(
                f"unsupported model: {config['pretrained']}; need to ensure correct tokenizers will be used before arbitrary hgf models are supported"
            )

        self.transformer_layer_1 = TFBertLayer(self.bert.config)
        self.transformer_layer_2 = TFBertLayer(self.bert.config)
        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.bert.config.hidden_size,))

        if config["aggregation"] == "maxp":
            self.aggregation = self.aggregate_using_maxp
        elif config["aggregation"] == "transformer":
            self.aggregation = self.aggregate_using_transformer
            input_embeddings = self.bert.get_input_embeddings()
            cls_token_id = tf.convert_to_tensor([101])
            cls_token_id = tf.reshape(cls_token_id, [1, 1])
            self.initial_cls_embedding = input_embeddings(input_ids=cls_token_id)
            self.initial_cls_embedding = tf.reshape(self.initial_cls_embedding, [1, self.bert.config.hidden_size])
            initializer = tf.random_normal_initializer(stddev=0.02)
            full_position_embeddings = tf.Variable(
                initial_value=initializer(shape=[self.num_passages + 1, self.bert.config.hidden_size]),
                name="passage_position_embedding",
            )
            self.full_position_embeddings = tf.expand_dims(full_position_embeddings, axis=0)

    def aggregate_using_maxp(self, cls):
        """
        cls has the shape [B, num_passages, hidden_size]
        """
        expanded_cls = tf.reshape(cls, [-1, self.num_passages, self.bert.config.hidden_size])
        aggregated = tf.reduce_max(expanded_cls, axis=1)

        return aggregated

    def aggregate_using_transformer(self, cls):
        expanded_cls = tf.reshape(cls, [-1, self.num_passages, self.bert.config.hidden_size])
        batch_size = tf.shape(expanded_cls)[0]
        tiled_initial_cls = tf.tile(self.initial_cls_embedding, multiples=[batch_size, 1])
        merged_cls = tf.concat((tf.expand_dims(tiled_initial_cls, axis=1), expanded_cls), axis=1)

        merged_cls += self.full_position_embeddings

        (transformer_out_1,) = self.transformer_layer_1(merged_cls, None, None, None)
        (transformer_out_2,) = self.transformer_layer_2(transformer_out_1, None, None, None)

        aggregated = transformer_out_2[:, 0, :]
        return aggregated

    def call(self, x, **kwargs):
        doc_input, doc_mask, doc_seg = x[0], x[1], x[2]
        batch_size = tf.shape(doc_input)[0]

        doc_input = tf.reshape(doc_input, [batch_size * self.num_passages, self.maxseqlen])
        doc_mask = tf.reshape(doc_mask, [batch_size * self.num_passages, self.maxseqlen])
        doc_seg = tf.reshape(doc_seg, [batch_size * self.num_passages, self.maxseqlen])

        cls = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 0, :]
        aggregated = self.aggregation(cls)

        return self.linear(aggregated)

    def predict_step(self, data):
        """
        Scores each passage and applies max pooling over it.
        """
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
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
        ConfigOption(
            "pretrained", "bert-base-uncased", "Pretrained model: bert-base-uncased, bert-base-msmarco, or electra-base-msmarco"
        ),
        ConfigOption("aggregation", "transformer"),
    ]

    def build_model(self):
        self.model = TFParade_Class(self.extractor, self.config)
        return self.model
