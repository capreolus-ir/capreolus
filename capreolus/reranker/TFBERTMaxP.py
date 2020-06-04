import tensorflow as tf
from transformers import TFBertForSequenceClassification

from capreolus.registry import Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class TFBERTMaxP_Class(tf.keras.Model):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.clsidx = extractor.clsidx  # The index of the CLS token
        self.sepidx = extractor.sepidx  # The index of the SEP token
        self.extractor = extractor
        self.bert = TFBertForSequenceClassification.from_pretrained(config["pretrained"])
        self.config = config
        self.aggregate_fn = self.get_aggregate_fn()

    def get_aggregate_fn(self):
        if self.config["mode"] == "maxp":
            return tf.math.reduce_max

    def call(self, x, **kwargs):
        posdoc_input, posdoc_mask, posdoc_seg, negdoc_input, negdoc_mask, negdoc_seg = x

        batch_size = tf.shape(posdoc_input)[0]
        num_passages = self.extractor.cfg["num_passages"]
        maxseqlen = self.extractor.cfg["maxseqlen"]

        posdoc_input = tf.reshape(posdoc_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        pos_passage_scores = self.bert(posdoc_input, attention_mask=posdoc_mask, token_type_ids=posdoc_seg)[0][:, 0]
        pos_passage_scores = tf.reshape(pos_passage_scores, [batch_size, num_passages])
        posdoc_score = tf.math.reduce_max(pos_passage_scores, axis=1)

        negdoc_input = tf.reshape(negdoc_input, [batch_size * num_passages, maxseqlen])
        negdoc_mask = tf.reshape(negdoc_mask, [batch_size * num_passages, maxseqlen])
        negdoc_seg = tf.reshape(negdoc_seg, [batch_size * num_passages, maxseqlen])

        neg_passage_scores = self.bert(negdoc_input, attention_mask=negdoc_mask, token_type_ids=negdoc_seg)[0][:, 0]
        neg_passage_scores = tf.reshape(neg_passage_scores, [batch_size, num_passages])
        negdoc_score = tf.math.reduce_max(neg_passage_scores, axis=1)

        return tf.stack([posdoc_scores, negdoc_scores], axis=1)

    # def call(self, x, **kwargs):
    #     pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]
    #     batch_size = tf.shape(pos_toks)[0]
    #     doclen = self.extractor.cfg["maxdoclen"]
    #     qlen = self.extractor.cfg["maxqlen"]
    #
    #     cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
    #     sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
    #     sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
    #     ones = tf.ones([batch_size, 1], dtype=tf.int64)
    #
    #     passagelen = self.config["passagelen"]
    #     stride = self.config["stride"]
    #     # TODO: Integer division would mean that we round down - the last passage would be lost
    #     num_passages = (doclen - passagelen) // stride
    #     # The passage level scores will be stored in these arrays
    #     pos_passage_scores = tf.TensorArray(tf.float32, size=doclen // passagelen, dynamic_size=False)
    #     neg_passage_scores = tf.TensorArray(tf.float32, size=doclen // passagelen, dynamic_size=False)
    #
    #     i = 0
    #     idx = 0
    #
    #     while idx < num_passages:
    #         # Get a passage and the corresponding mask
    #         pos_passage = pos_toks[:, i : i + passagelen]
    #         pos_passage_mask = posdoc_mask[:, i : i + passagelen]
    #         neg_passage = neg_toks[:, i : i + passagelen]
    #         neg_passage_mask = negdoc_mask[:, i : i + passagelen]
    #
    #         # Prepare the input to bert
    #         query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
    #         query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
    #         query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
    #         query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
    #         query_passage_segments_tensor = tf.concat(
    #             [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1
    #         )
    #
    #         # Actual bert scoring
    #         pos_passage_score = self.bert(
    #             query_pos_passage_tokens_tensor,
    #             attention_mask=query_pos_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         neg_passage_score = self.bert(
    #             query_neg_passage_tokens_tensor,
    #             attention_mask=query_neg_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         pos_passage_scores = pos_passage_scores.write(idx, pos_passage_score)
    #         neg_passage_scores = neg_passage_scores.write(idx, neg_passage_score)
    #
    #         idx += 1
    #         i += stride
    #
    #     posdoc_scores = tf.math.reduce_max(pos_passage_scores.stack(), axis=0)
    #     negdoc_scores = tf.math.reduce_max(neg_passage_scores.stack(), axis=0)
    #     return tf.stack([posdoc_scores, negdoc_scores], axis=1)


class TFBERTMaxP(Reranker):
    name = "TFBERTMaxP"
    dependencies = {
        "extractor": Dependency(module="extractor", name="bertpassage"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        pretrained = "bert-base-uncased"
        passagelen = 100
        stride = 20
        mode = "maxp"

    def build(self):
        self.model = TFBERTMaxP_Class(self["extractor"], self.cfg)
        return self.model
