import copy

import tensorflow as tf
from transformers import TFBertModel
from transformers.modeling_tf_bert import TFBertLayer

from capreolus.registry import Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class TFDocuBERT_Class(tf.keras.Model):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFDocuBERT_Class, self).__init__(*args, **kwargs)
        self.config = config
        self.clsidx = extractor.clsidx  # The index of the CLS token
        self.sepidx = extractor.sepidx  # The index of the SEP token
        self.extractor = extractor
        self.bert = TFBertModel.from_pretrained(config["pretrained"])
        self.transformer_layer_1 = TFBertLayer(self.bert.config)
        self.transformer_layer_2 = TFBertLayer(self.bert.config)
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.config["numpassages"] + 1, self.bert.config.hidden_size))

    @tf.function
    def call(self, x, **kwargs):
        pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]

        batch_size = tf.shape(pos_toks)[0]
        num_passages = self.config["numpassages"]
        stride = self.config["stride"]
        passagelen = self.config["passagelen"]
        qlen = self.extractor.cfg["maxqlen"]

        # CLS and SEP tokens of shape (batch_size, 1)
        cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
        sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
        sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
        ones = tf.ones([batch_size, 1], dtype=tf.int64)

        # Get the [CLS] token embedding, and add it to a list
        intial_cls_embedding = tf.gather(self.bert.get_input_embeddings().word_embeddings, [self.clsidx])
        pos_passage_scores = tf.TensorArray(tf.float32, size=num_passages + 1, dynamic_size=False)
        pos_passage_scores = pos_passage_scores.write(0, intial_cls_embedding)
        neg_passage_scores = tf.TensorArray(tf.float32, size=num_passages + 1, dynamic_size=False)
        neg_passage_scores = neg_passage_scores.write(0, intial_cls_embedding)

        idx = 0
        while idx < num_passages:
            i = idx * stride
            # Get a passage and the corresponding mask
            pos_passage = pos_toks[:, i : i + passagelen]
            pos_passage_mask = posdoc_mask[:, i : i + passagelen]
            neg_passage = neg_toks[:, i : i + passagelen]
            neg_passage_mask = negdoc_mask[:, i : i + passagelen]

            # Prepare the input to bert
            query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
            query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
            query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
            query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
            query_passage_segments_tensor = tf.concat(
                [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1
            )

            # Actual bert scoring
            pos_passage_score = self.bert(
                query_pos_passage_tokens_tensor,
                attention_mask=query_pos_passage_mask,
                token_type_ids=query_passage_segments_tensor,
            )[0][:, 0]
            neg_passage_score = self.bert(
                query_neg_passage_tokens_tensor,
                attention_mask=query_neg_passage_mask,
                token_type_ids=query_passage_segments_tensor,
            )[0][:, 0]
            pos_passage_scores = pos_passage_scores.write(idx, pos_passage_score)
            neg_passage_scores = neg_passage_scores.write(idx, neg_passage_score)

            idx += 1

        logger.info("cls_token_embeddings array shape is {}".format(pos_passage_scores.stack()))
        pos_layer_out_1, pos_attn_out_1 = self.transformer_layer_1(pos_passage_scores.stack())
        pos_layer_out_2, pos_attn_out_2 = self.transformer_layer_2(pos_layer_out_1)

        neg_layer_out_1, neg_attn_out_1 = self.transformer_layer_1(neg_passage_scores.stack())
        neg_layer_out_2, neg_attn_out_2 = self.transformer_layer_2(neg_layer_out_1)

        logger.info("Final hstates shape is {}".format(pos_layer_out_2))

        pos_final_cls_embedding = pos_layer_out_2[:, 0]
        neg_final_cls_embedding = neg_layer_out_2[:, 0]

        pos_score = self.linear(pos_final_cls_embedding)
        neg_score = self.linear(neg_final_cls_embedding)

        return tf.stack([pos_score, neg_score], axis=1)


class TFDocuBERT(Reranker):
    name = "TFDocuBERT"
    dependencies = {
        "extractor": Dependency(module="extractor", name="berttext"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        pretrained = "bert-base-uncased"

        # Corresponding to maxdoclen of 800. Original paper uses 16 passages
        # TODO: Talk to Canjia and see if maxdoclen should be 950
        numpassages = 13
        passagelen = 150
        stride = 50
        mode = "transformer"

    def build(self):
        self.model = TFDocuBERT_Class(self["extractor"], self.cfg)
        return self.model
