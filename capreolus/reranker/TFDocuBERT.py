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
        self.extractor = extractor
        self.bert = TFBertModel.from_pretrained(config["pretrained"])
        self.transformer_layer_1 = TFBertLayer(self.bert.config)
        self.transformer_layer_2 = TFBertLayer(self.bert.config)
        # self.num_passages = (self.extractor.cfg["maxdoclen"] - config["passagelen"]) // self.config["stride"]
        self.num_passages = extractor.cfg["numpassages"]
        self.passagelen = extractor.cfg["passagelen"]
        self.linear = tf.keras.layers.Dense(1, input_shape=(self.bert.config.hidden_size, ))

    # def call(self, x, **kwargs):
    #     pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]
    #
    #     batch_size = tf.shape(pos_toks)[0]
    #     num_passages = self.num_passages
    #     stride = self.config["stride"]
    #     passagelen = self.config["passagelen"]
    #     qlen = self.extractor.cfg["maxqlen"]
    #
    #     # CLS and SEP tokens of shape (batch_size, 1)
    #     cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
    #     sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
    #     sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
    #     ones = tf.ones([batch_size, 1], dtype=tf.int64)
    #
    #     # Get the [CLS] token embedding, and add it to a list
    #     single_cls_embedding = tf.gather(self.bert.get_input_embeddings().word_embeddings, [self.clsidx])
    #     cls_embedding_batch = tf.tile(single_cls_embedding, [batch_size] + [1 for x in range(1, len(single_cls_embedding.shape))])
    #     pos_passage_cls_list = tf.TensorArray(tf.float32, size=num_passages + 1, dynamic_size=False)
    #     pos_passage_cls_list = pos_passage_cls_list.write(0, cls_embedding_batch)
    #     neg_passage_cls_list = tf.TensorArray(tf.float32, size=num_passages + 1, dynamic_size=False)
    #     neg_passage_cls_list = neg_passage_cls_list.write(0, cls_embedding_batch)
    #
    #     # Get the [CLS] embedding corresponding to each passage in the doc and add it to the list
    #     idx = 0
    #     while idx < num_passages:
    #         i = idx * stride
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
    #         pos_passage_cls = self.bert(
    #             query_pos_passage_tokens_tensor,
    #             attention_mask=query_pos_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         neg_passage_cls = self.bert(
    #             query_neg_passage_tokens_tensor,
    #             attention_mask=query_neg_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         pos_passage_cls_list = pos_passage_cls_list.write(idx, pos_passage_cls)
    #         neg_passage_cls_list = neg_passage_cls_list.write(idx, neg_passage_cls)
    #
    #         idx += 1
    #
    #     # pos_passage_cls_list.stack() gives tensor of shape (num_passages, batch_size, hidden_size)
    #     # We need to permute it to (batch_size, num_passages, hidden_size)
    #     # self.transformer_layer_1 also requires attention_mask and head_mask, which are explicitly passed as None here
    #     pos_layer_out_1, = self.transformer_layer_1((tf.transpose(pos_passage_cls_list.stack(), perm=[1, 0, 2]), None, None))
    #     pos_layer_out_2, = self.transformer_layer_2((pos_layer_out_1, None, None))
    #
    #     neg_layer_out_1, = self.transformer_layer_1((tf.transpose(neg_passage_cls_list.stack(), perm=[1, 0, 2]), None, None))
    #     neg_layer_out_2, = self.transformer_layer_2((neg_layer_out_1, None, None))
    #
    #     # Obtain the [CLS] embedding of transformer_layer_2 (i.e the hidden state corresponding to the first element
    #     # in the input sequence) and reshape it to get a a tensor of shape (batch_size, hidden_size)
    #     pos_final_cls_embedding = tf.reshape(pos_layer_out_2[:, 0], [batch_size, self.bert.config.hidden_size])
    #     neg_final_cls_embedding = tf.reshape(neg_layer_out_2[:, 0], [batch_size, self.bert.config.hidden_size])
    #
    #     # Obtain logits of the shape [batch_size]
    #     pos_score = tf.reshape(self.linear(pos_final_cls_embedding), [batch_size])
    #     neg_score = tf.reshape(self.linear(neg_final_cls_embedding), [batch_size])
    #
    #     return tf.stack([pos_score, neg_score], axis=1)
    @tf.function
    def call(self, x, **kwargs):
        posdoc_input, posdoc_mask, posdoc_seg, negdoc_input, negdoc_mask, negdoc_seg = x
        batch_size = tf.shape(posdoc_input)[0]

        # Reshape to (batch_size * num_passages, passagelen)
        posdoc_input = tf.reshape(posdoc_input, [batch_size * self.num_passages, self.passagelen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * self.num_passages, self.passagelen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * self.num_passages, self.passagelen])

        pos_cls = self.bert(posdoc_input, attention_mask=posdoc_mask, token_type_ids=posdoc_seg)[0][:, 0]
        pos_cls = tf.reshape(pos_cls, [batch_size, self.num_passages, self.bert.config.hidden_size])

        pos_transformer_out1, = self.transformer_layer_1((pos_cls, None, None))
        pos_transformer_out2, = self.transformer_layer_2((pos_transformer_out1, None, None))
        pos_final_cls = tf.reshape(pos_transformer_out2[:, 0], [batch_size, self.bert.config.hidden_size])

        pos_score = tf.reshape(self.linear(pos_final_cls), [batch_size])

        def get_neg_score(negdoc_input, negdoc_mask, negdoc_seg):
            batch_size = tf.shape(negdoc_input)[0]
            negdoc_input = tf.reshape(negdoc_input, [batch_size * self.num_passages, self.passagelen])
            negdoc_mask = tf.reshape(negdoc_mask, [batch_size * self.num_passages, self.passagelen])
            negdoc_seg = tf.reshape(negdoc_seg, [batch_size * self.num_passages, self.passagelen])

            neg_cls = self.bert(negdoc_input, attention_mask=negdoc_mask, token_type_ids=negdoc_seg)[0][:, 0]
            neg_cls = tf.reshape(neg_cls, [batch_size, self.num_passages, self.bert.config.hidden_size])

            neg_transformer_out1, = self.transformer_layer_1((neg_cls, None, None))
            neg_transformer_out2, = self.transformer_layer_2((neg_transformer_out1, None, None))
            neg_final_cls = tf.reshape(neg_transformer_out2[:, 0], [batch_size, self.bert.config.hidden_size])

            neg_score = tf.reshape(self.linear(neg_final_cls), [batch_size])

            return neg_score

        def get_fake_neg_score():
            # Saves an awful lot of trouble by not passing a zero tensor through BERT
            return tf.zeros((batch_size))

        neg_score = tf.cond(tf.math.equal(tf.math.count_nonzero(negdoc_input), 0), false_fn=lambda: get_neg_score(negdoc_input, negdoc_mask, negdoc_seg), true_fn=get_fake_neg_score)

        return tf.stack([pos_score, neg_score], axis=1)



class TFDocuBERT(Reranker):
    name = "TFDocuBERT"
    dependencies = {
        "extractor": Dependency(module="extractor", name="bertpassage"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        pretrained = "bert-base-uncased"

        mode = "transformer"

    def build(self):
        self.model = TFDocuBERT_Class(self["extractor"], self.cfg)
        return self.model
