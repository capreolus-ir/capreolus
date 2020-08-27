import tensorflow as tf
from transformers import TFBertForSequenceClassification

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class TFVanillaBert_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFVanillaBert_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor

        # TFBertForSequenceClassification contains both the BERT and the linear classifier layers
        self.bert = TFBertForSequenceClassification.from_pretrained(config["pretrained"], hidden_dropout_prob=0.1)

        assert extractor.config["numpassages"] == 1, "numpassages should be 1 for TFVanillaBERT"
        self.config = config

    def call(self, x, **kwargs):
        """
        Returns logits of shape [2]
        """
        doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]
        doc_scores = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0]

        return doc_scores

    def predict_step(self, data):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = tf.shape(posdoc_bert_input)[1]
        tf.debugging.assert_equal(num_passages, 1)
        maxseqlen = self.extractor.config["maxseqlen"]

        posdoc_bert_input = tf.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        doc_scores = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), training=False)[:, 1]
        tf.debugging.assert_equal(tf.shape(doc_scores), [batch_size])

        return doc_scores

    def score(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        return self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)

    def score_pair(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        pos_score = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)[:, 1]
        neg_score = self.call((negdoc_bert_input, negdoc_mask, negdoc_seg), **kwargs)[:, 1]

        return pos_score, neg_score


@Reranker.register
class TFVanillaBERT(Reranker):
    """
    TensorFlow implementation of Vanilla BERT.
    Input is of the form [CLS] sentence A [SEP] sentence B [SEP]
    The "score" of a query (sentence A) - document (sentence B) pair is the probability that the document is relevant
    to the query. This is achieved through a linear classifier layer attached to BERT's last layer and using the logits[1] as the score.
    """

    module_name = "TFVanillaBERT"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [ConfigOption("pretrained", "bert-base-uncased", "pretrained model to load")]

    def build_model(self):
        self.model = TFVanillaBert_Class(self.extractor, self.config)
        return self.model
