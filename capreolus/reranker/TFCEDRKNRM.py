import math

import tensorflow as tf
from transformers import TFAutoModel

from capreolus import ConfigOption, Dependency, get_logger
from capreolus.reranker import Reranker
from capreolus.reranker.common import NewRbfKernelBankTF, new_similarity_matrix_tf

logger = get_logger(__name__)


class TFCEDRKNRM_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config

        if config["pretrained"] == "electra-base-msmarco":
            self.bert = TFAutoModel.from_pretrained(
                "Capreolus/electra-base-msmarco", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "electra-base":
            self.bert = TFAutoModel.from_pretrained(
                "google/electra-base-discriminator", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = TFAutoModel.from_pretrained(
                "Capreolus/bert-base-msmarco", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )
        elif config["pretrained"] == "bert-base-uncased":
            self.bert = TFAutoModel.from_pretrained(
                "bert-base-uncased", hidden_dropout_prob=config["hidden_dropout_prob"], output_hidden_states=True
            )

        self.hidden_size = self.bert.config.hidden_size
        mus = list(self.config["mus"]) + [1.0]
        sigmas = [self.config["sigma"] for _ in self.config["mus"]] + [0.01]
        logger.debug("mus: %s", mus)
        self.kernels = NewRbfKernelBankTF(mus, sigmas, dim=1, requires_grad=self.config["gradkernels"])

        if -1 in self.config["simmat_layers"]:
            assert len(self.config["simmat_layers"]) == 1
            assert self.config["cls"] is not None
            self._compute_simmat = False
            combine_size = 0
        else:
            self._compute_simmat = True
            combine_size = self.kernels.count() * len(self.config["simmat_layers"])

        assert self.config["cls"] in ("avg", "max", None)
        if self.config["cls"]:
            combine_size += self.hidden_size

        # use weight init from PyTorch 0.4
        if config["combine_hidden"] == 0:
            stdv = 1.0 / math.sqrt(combine_size)
            weight_init = tf.keras.initializers.RandomUniform(minval=-stdv, maxval=stdv)
            combine_steps = [
                tf.keras.layers.Dense(1, input_shape=(combine_size,), kernel_initializer=weight_init, dtype=tf.float32)
            ]
        else:
            stdv1 = 1.0 / math.sqrt(combine_size)
            weight_init1 = tf.keras.initializers.RandomUniform(minval=-stdv1, maxval=stdv1)
            stdv2 = 1.0 / math.sqrt(config["combine_hidden"])
            weight_init2 = tf.keras.initializers.RandomUniform(minval=-stdv2, maxval=stdv2)
            combine_steps = [
                tf.keras.layers.Dense(
                    config["combine_hidden"], input_shape=(combine_size,), kernel_initializer=weight_init1, dtype=tf.float32
                ),
                tf.keras.layers.Dense(
                    1, input_shape=(config["combine_hidden"],), kernel_initializer=weight_init2, dtype=tf.float32
                ),
            ]

        self.combine = tf.keras.Sequential(combine_steps)

        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        # TODO we include SEP in maxqlen due to the way the simmat is constructed... (and another SEP in document)
        # (maxqlen is the actual query length and does not count CLS or SEP)
        self.maxqlen = extractor.config["maxqlen"] + 1
        # decreased by 1 because we remove CLS before generating embeddings
        self.maxdoclen = self.maxseqlen - 1

    def masked_simmats(self, embeddings, bert_mask, bert_segments):
        bert_mask = tf.cast(bert_mask, embeddings.dtype)
        # segment 0 contains '[CLS] query [SEP]' and segment 1 contains 'document [SEP]'
        query_mask = bert_mask * tf.cast(bert_segments == 0, bert_mask.dtype)
        query_mask = tf.expand_dims(query_mask, axis=-1)
        padded_query = (query_mask * embeddings)[:, : self.maxqlen]
        query_mask = query_mask[:, : self.maxqlen]

        doc_mask = bert_mask * tf.cast(bert_segments == 1, bert_mask.dtype)
        doc_mask = tf.expand_dims(doc_mask, axis=-1)
        # padded_doc length is maxsdoclen; zero padding on  both left and right of doc
        padded_doc = doc_mask * embeddings

        # (maxqlen, maxdoclen)
        simmat = new_similarity_matrix_tf(padded_query, padded_doc, query_mask, doc_mask, 0)
        return simmat, doc_mask, query_mask

    def knrm(self, bert_output, bert_mask, bert_segments, batch_size):
        # create similarity matrix for each passage (skipping CLS)
        passage_simmats, passage_doc_mask, passage_query_mask = self.masked_simmats(
            bert_output[:, 1:], bert_mask[:, 1:], bert_segments[:, 1:]
        )

        passage_simmats = tf.reshape(passage_simmats, [batch_size, self.num_passages, self.maxqlen, self.maxdoclen])
        passage_doc_mask = tf.reshape(passage_doc_mask, [batch_size, self.num_passages, 1, -1])

        # concat similarity matrices along document dimension; query mask is the same across passages
        doc_simmat = tf.concat([passage_simmats[:, PIDX, :, :] for PIDX in range(self.num_passages)], axis=2)
        doc_mask = tf.concat([passage_doc_mask[:, PIDX, :, :] for PIDX in range(self.num_passages)], axis=2)
        query_mask = tf.reshape(passage_query_mask, [batch_size, self.num_passages, -1, 1])[:, 0, :, :]

        # KNRM on similarity matrix
        prepooled_doc = self.kernels(doc_simmat)
        prepooled_doc = (
            prepooled_doc * tf.reshape(doc_mask, [batch_size, 1, 1, -1]) * tf.reshape(query_mask, [batch_size, 1, -1, 1])
        )

        # sum over document
        knrm_features = tf.reduce_sum(prepooled_doc, axis=3)
        knrm_features = tf.math.log(tf.maximum(knrm_features, 1e-6)) * 0.01
        # sum over query
        knrm_features = tf.reduce_sum(knrm_features, axis=2)

        return knrm_features

    def call(self, x, **kwargs):
        doc_input, doc_mask, doc_seg = x[0], x[1], x[2]
        batch_size = tf.shape(doc_input)[0]

        doc_input = tf.reshape(doc_input, [batch_size * self.num_passages, self.maxseqlen])
        doc_mask = tf.reshape(doc_mask, [batch_size * self.num_passages, self.maxseqlen])
        doc_seg = tf.reshape(doc_seg, [batch_size * self.num_passages, self.maxseqlen])

        # get BERT embeddings (including CLS) for each passage
        # TODO switch to hgf's ModelOutput after bumping tranformers version
        outputs = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)
        if self.config["pretrained"].startswith("bert-"):
            outputs = (outputs[0], outputs[2])
        bert_output, all_layer_output = outputs

        #  embeddings to create the CLS feature
        cls = bert_output[:, 0, :]
        if self.config["cls"] == "max":
            cls_features = tf.reshape(cls, [batch_size, self.num_passages, self.hidden_size])
            cls_features = tf.reduce_max(cls_features, axis=1)
        elif self.config["cls"] == "avg":
            cls_features = tf.reshape(cls, [batch_size, self.num_passages, self.hidden_size])
            cls_features = tf.reduce_mean(cls_features, axis=1)

        # create KNRM features for each output layer
        if self._compute_simmat:
            layer_knrm_features = [
                self.knrm(all_layer_output[LIDX], doc_mask, doc_seg, batch_size) for LIDX in self.config["simmat_layers"]
            ]

        # concat CLS+KNRM features and pass to linear layer
        if self.config["cls"] and self._compute_simmat:
            all_features = tf.concat([cls_features] + layer_knrm_features, axis=1)
        elif self._compute_simmat:
            all_features = tf.concat(layer_knrm_features, axis=1)
        elif self.config["cls"]:
            all_features = cls_features
        else:
            raise ValueError("invalid config: %s" % self.config)

        score = self.combine(all_features)
        return score

    def predict_step(self, data):
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
class TFCEDRKNRM(Reranker):
    module_name = "TFCEDRKNRM"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="pooledbertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [
        ConfigOption(
            "pretrained",
            "electra-base",
            "Pretrained model: bert-base-uncased, bert-base-msmarco, electra-base, or electra-base-msmarco",
        ),
        ConfigOption("mus", [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9], "mus", value_type="floatlist"),
        ConfigOption("sigma", 0.1, "sigma"),
        ConfigOption("gradkernels", True, "tune mus and sigmas"),
        ConfigOption("hidden_dropout_prob", 0.1, "The dropout probability of BERT-like model's hidden layers."),
        ConfigOption("simmat_layers", "0..12,1", "Layer outputs to include in similarity matrix", value_type="intlist"),
        ConfigOption("combine_hidden", 1024, "Hidden size to use with combination FC layer (0 to disable)"),
        ConfigOption("cls", "avg", "Handling of CLS token: avg, max, or None"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            with self.trainer.strategy.scope():
                self.model = TFCEDRKNRM_Class(self.extractor, self.config)
        return self.model
