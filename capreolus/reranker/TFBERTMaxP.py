import tensorflow as tf
from transformers import TFBertForSequenceClassification

from capreolus.registry import Dependency
from capreolus.reranker import Reranker


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
        """
        During training, both posdoc and negdoc are passed
        During eval, both posdoc and negdoc are passed but negdoc would be a zero tensor
        Whether negdoc is a legit doc tensor or a dummy zero tensor is determined by which sampler is used
        (eg: sampler.TrainDataset) as well as the extractor (eg: EmbedText)
        """

        pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]
        batch_size = tf.shape(pos_toks)[0]
        doclen = tf.shape(pos_toks)[1]
        qlen = tf.shape(query_toks)[1]

        cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
        sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
        sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
        ones = tf.ones([batch_size, 1], dtype=tf.int64)

        passagelen = self.config["passagelen"]
        # overlap = self.config["overlap"]
        pos_passage_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        neg_passage_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True)


        # def condition(idx, ta1, ta2):
        #     return tf.less(idx * (passagelen-overlap), doclen)
        #
        # idx = tf.constant(0)
        # loop_vars = (idx, pos_passage_scores, neg_passage_scores)
        #
        # def body(_idx, _pos_passage_scores, _neg_passage_scores):
        #     i = _idx * (passagelen - overlap)
        #     pos_passage = pos_toks[:, i: i+passagelen]
        #     pos_passage_mask = posdoc_mask[:, i: i+passagelen]
        #     neg_passage = neg_toks[:, i:i+passagelen]
        #     neg_passage_mask = negdoc_mask[:, i: i+passagelen]
        #
        #     query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
        #     query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
        #     query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
        #     query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
        #     query_passage_segments_tensor = tf.concat([tf.zeros([batch_size, qlen+2]), tf.ones([batch_size, passagelen + 1])], axis=1)
        #     pos_passage_score = self.bert(
        #         query_pos_passage_tokens_tensor, attention_mask=query_pos_passage_mask, token_type_ids=query_passage_segments_tensor
        #     )[0][:, 0]
        #     neg_passage_score = self.bert(
        #         query_neg_passage_tokens_tensor, attention_mask=query_neg_passage_mask, token_type_ids=query_passage_segments_tensor
        #     )[0][:, 0]
        #     _pos_passage_scores = _pos_passage_scores.write(idx, tf.reshape(pos_passage_score, [batch_size]))
        #     _neg_passage_scores = _neg_passage_scores.write(idx, tf.reshape(neg_passage_score, [batch_size]))
        #
        #     return (tf.add(_idx, 1), _pos_passage_scores, _neg_passage_scores)
        #
        # tf.while_loop(condition, body, loop_vars)
        #
        # pos_passage_scores = tf.reshape(pos_passage_scores.stack(), [batch_size, -1])
        # neg_passage_scores = tf.reshape(neg_passage_scores.stack(), [batch_size, -1])
        # posdoc_score = tf.math.reduce_max(pos_passage_scores, axis=1)
        # negdoc_score = tf.math.reduce_max(neg_passage_scores, axis=1)
        i = 0
        pos_passage = pos_toks[:, i: i + passagelen]
        pos_passage_mask = posdoc_mask[:, i: i + passagelen]
        neg_passage = neg_toks[:, i:i + passagelen]
        neg_passage_mask = negdoc_mask[:, i: i + passagelen]

        query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
        query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
        query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
        query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
        query_passage_segments_tensor = tf.concat(
            [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1)
        pos_passage_score = self.bert(
            query_pos_passage_tokens_tensor, attention_mask=query_pos_passage_mask,
            token_type_ids=query_passage_segments_tensor
        )[0][:, 0]
        neg_passage_score = self.bert(
            query_neg_passage_tokens_tensor, attention_mask=query_neg_passage_mask,
            token_type_ids=query_passage_segments_tensor
        )[0][:, 0]

        pos_passage_scores = pos_passage_scores.write(0, pos_passage_score)
        neg_passage_scores = neg_passage_scores.write(0, neg_passage_score)

        i = passagelen
        pos_passage = pos_toks[:, i: i + passagelen]
        pos_passage_mask = posdoc_mask[:, i: i + passagelen]
        neg_passage = neg_toks[:, i:i + passagelen]
        neg_passage_mask = negdoc_mask[:, i: i + passagelen]

        query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
        query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
        query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
        query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
        query_passage_segments_tensor = tf.concat(
            [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1)
        pos_passage_score = self.bert(
            query_pos_passage_tokens_tensor, attention_mask=query_pos_passage_mask,
            token_type_ids=query_passage_segments_tensor
        )[0][:, 0]
        neg_passage_score = self.bert(
            query_neg_passage_tokens_tensor, attention_mask=query_neg_passage_mask,
            token_type_ids=query_passage_segments_tensor
        )[0][:, 0]

        pos_passage_scores = pos_passage_scores.write(1, pos_passage_score)
        neg_passage_scores = neg_passage_scores.write(1, neg_passage_score)

        posdoc_scores = tf.math.reduce_max(tf.reshape(pos_passage_scores.stack(), [batch_size, -1]), axis=1)
        negdoc_scores = tf.math.reduce_max(tf.reshape(neg_passage_scores.stack(), [batch_size, -1]), axis=1)
        return tf.stack([posdoc_scores, negdoc_scores], axis=1)


class TFBERTMaxP(Reranker):
    name = "TFBERTMaxP"
    dependencies = {
        "extractor": Dependency(module="extractor", name="berttext"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        pretrained = "bert-base-uncased"
        passagelen = 80
        overlap = 20
        mode = "maxp"

    def build(self):
        self.model = TFBERTMaxP_Class(self["extractor"], self.cfg)
        return self.model