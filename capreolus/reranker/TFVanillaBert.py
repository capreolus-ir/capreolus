import tensorflow as tf
from transformers import TFBertForSequenceClassification

from capreolus import ConfigOption, Dependency
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

        query_posdoc_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_toks, sep_2], axis=1)
        query_negdoc_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_toks, sep_2], axis=1)
        ones = tf.ones([batch_size, 1], dtype=tf.int64)
        query_posdoc_mask = tf.concat([ones, query_mask, ones, posdoc_mask, ones], axis=1)
        query_negdoc_mask = tf.concat([ones, query_mask, ones, negdoc_mask, ones], axis=1)
        query_doc_segments_tensor = tf.concat([tf.zeros([batch_size, qlen + 2]), tf.zeros([batch_size, doclen + 1])], axis=1)
        posdoc_score = self.bert(
            query_posdoc_tokens_tensor, attention_mask=query_posdoc_mask, token_type_ids=query_doc_segments_tensor
        )[0][:, 0]
        negdoc_score = self.bert(
            query_negdoc_tokens_tensor, attention_mask=query_negdoc_mask, token_type_ids=query_doc_segments_tensor
        )[0][:, 0]

        # TODO: Verify that negdoc_score is indeed always zero whenever a zero negdoc tensor is passed into it
        return tf.stack([posdoc_score, negdoc_score], axis=1)


@Reranker.register
class TFVanillaBERT(Reranker):
    """TensorFlow implementation of Vanilla BERT."""

    module_name = "TFVanillaBERT"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="berttext"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [ConfigOption("pretrained", "bert-base-uncased", "pretrained model to load")]

    def build_model(self):
        self.model = TFVanillaBert_Class(self.extractor, self.config)
        return self.model
