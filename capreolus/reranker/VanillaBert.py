import tensorflow as tf
from transformers import TFBertForSequenceClassification

from capreolus.registry import Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class TFVanillaBert_Class(tf.keras.Model):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFVanillaBert_Class, self).__init__(*args, **kwargs)
        self.clsidx = extractor.clsidx  # The index of the CLS token
        self.sepidx = extractor.sepidx  # The index of the SEP token
        self.extractor = extractor
        self.bert = TFBertForSequenceClassification.from_pretrained(config["pretrained"])
        self.config = config

    def call(self, x, **kwargs):
        pos_toks, neg_toks, query_toks, query_idf = x[0], x[1], x[2], x[3]
        pos_toks = tf.cast(pos_toks, tf.int32)
        neg_toks = tf.cast(neg_toks, tf.int32)
        query_toks = tf.cast(query_toks, tf.int32)
        batch_size = tf.shape(pos_toks)[0]
        doclen = tf.shape(pos_toks)[1]
        qlen = tf.shape(query_toks)[1]

        cls = tf.fill([batch_size, 1], self.clsidx, name="clstoken")
        sep_1 = tf.fill([batch_size, 1], self.sepidx, name="septoken1")
        sep_2 = tf.fill([batch_size, 1], self.sepidx, name="septoken2")

        query_posdoc_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_toks, sep_2], axis=1)
        query_negdoc_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_toks, sep_2], axis=1)
        query_doc_segments_tensor = tf.concat([tf.zeros([batch_size, qlen + 2]), tf.zeros([batch_size, doclen + 1])], axis=1)
        posdoc_score = self.bert(query_posdoc_tokens_tensor, token_type_ids=query_doc_segments_tensor)[0][:, 0]
        negdoc_score = self.bert(query_negdoc_tokens_tensor, token_type_ids=query_doc_segments_tensor)[0][:, 0]

        # TODO: Verify that negdoc_score is indeed always zero whenever a zero negdoc tensor is passed into it
        return posdoc_score - negdoc_score


class TFVanillaBERT(Reranker):
    name = "TFVanillaBERT"
    dependencies = {
        "extractor": Dependency(module="extractor", name="bert"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        pretrained = "bert-base-uncased"

    def build(self):
        self.model = TFVanillaBert_Class(self["extractor"], self.cfg)
        return self.model
