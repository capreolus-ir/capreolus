import copy

import tensorflow as tf
from transformers import TFBertModel
from transformers.modeling_tf_bert import TFBertEncoder

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
        duplicate_config = copy.copy(self.bert.config)
        duplicate_config.num_hidden_layers = 2
        self.transformer_layers = TFBertEncoder(duplicate_config)
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.config["numpassages"] + 1, self.bert.config.hidden_size))
        self.aggregate_fn = self.get_aggregate_fn()

    def get_aggregate_fn(self):
        if self.config["mode"] == "maxp":
            return tf.math.reduce_max

    def get_passage_cls_embedding(self, i, doc_toks, doc_mask, query_toks, query_mask, cls, sep_1, sep_2, ones):
        """
        Returns the score for the ith pos passage and the ith neg passage
        """
        batch_size = tf.shape(doc_toks)[0]
        stride = self.config["stride"]
        passagelen = self.config["passagelen"]
        qlen = self.config["maxqlen"]

        p_start = i * stride

        passage = doc_toks[:, p_start : p_start + passagelen]
        passage_mask = doc_mask[:, p_start : p_start + passagelen]

        # Prepare the input to bert
        query_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, passage, sep_2], axis=1)
        query_passage_mask = tf.concat([ones, query_mask, ones, passage_mask, ones], axis=1)
        query_passage_segments_tensor = tf.concat(
            [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1
        )

        # Actual bert scoring
        last_hidden_state, pooler_output = self.bert(
            query_passage_tokens_tensor, attention_mask=query_passage_mask, token_type_ids=query_passage_segments_tensor
        )[0][:, 0]

        return last_hidden_state[0]

    def get_doc_score(self, doc_toks, doc_mask, query_toks, query_mask):
        batch_size = tf.shape(doc_toks)[0]
        num_passages = self.config["numpassages"]

        # CLS and SEP tokens of shape (batch_size, 1)
        cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
        sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
        sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
        ones = tf.ones([batch_size, 1], dtype=tf.int64)

        # Get the [CLS] token embedding, and add it to a list
        intial_cls_embedding = tf.gather(self.bert.get_input_embeddings().word_embeddings, [self.clsidx])
        cls_token_embeddings = tf.TensorArray(tf.float32, size=num_passages + 1, dynamic_size=False)
        cls_token_embeddings = cls_token_embeddings.write(0, intial_cls_embedding)

        # Get the contextual [CLS] embedding for each passage in the doc and add it to a list
        i = 0
        while i < num_passages:
            cls_embedding = self.get_passage_cls_embedding(i, doc_toks, doc_mask, query_toks, query_mask, cls, sep_1, sep_2, ones)
            cls_token_embeddings = cls_token_embeddings.write(i + 1, cls_embedding)
            i += 1

        logger.info("cls_token_embeddings array shape is {}".format(cls_token_embeddings.stack()))
        final_hstates, all_hstates, all_att = self.transformer_layers(cls_token_embeddings.stack())
        logger.info("Final hstates shape is {}".format(final_hstates))
        final_cls_embedding = final_hstates[:, 0]

        score = self.linear(final_cls_embedding)

        return score

    def call(self, x, **kwargs):
        pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]
        posdoc_scores = self.get_doc_score(pos_toks, posdoc_mask, query_toks, query_mask)
        negdoc_scores = self.get_doc_score(neg_toks, negdoc_mask, query_toks, query_mask)

        return tf.stack([posdoc_scores, negdoc_scores], axis=1)


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
