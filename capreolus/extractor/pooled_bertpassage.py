import tensorflow as tf
import numpy as np

from capreolus import Dependency, ConfigOption, get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError
from . import Extractor
from .bertpassage import BertPassage

logger = get_logger(__name__)


@Extractor.register
class PooledBertPassage(BertPassage):
    """
    Extracts passages from the document to be later consumed by a BERT based model.
    Different from BertPassage in the sense that all the passages from a document "stick together" during training - the
    resulting feature always have the shape (batch, num_passages, maxseqlen) - and this allows the reranker to pool
    over passages from the same document during training

    """

    module_name = "pooledbertpassage"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]

    pad = 0
    pad_tok = "[PAD]"

    config_spec = [
        ConfigOption("maxseqlen", 256, "Maximum input length for BERT"),
        ConfigOption("usecache", False, "Should the extracted features be cached?"),
        ConfigOption("passagelen", 150, "Length of the extracted passage"),
        ConfigOption("stride", 100, "Stride"),
        ConfigOption("sentences", False, "Use a sentence tokenizer to form passages"),
        ConfigOption("numpassages", 16, "Number of passages per document"),
        ConfigOption(
            "prob",
            0.1,
            "The probability that a passage from the document will be used for training (the first passage is always used)",
        ),
    ]

    def create_tf_train_feature(self, sample):
        """
        Returns a set of features from a doc.
        Of the num_passages passages that are present in a document, we use only a subset of it.
        params:
        sample - A dict where each entry has the shape [batch_size, num_passages, maxseqlen]

        Returns a list of features. Each feature is a dict, and each value in the dict has the shape [batch_size, maxseqlen].
        Yes, the output shape is different to the input shape because we sample from the passages.
        """
        return self.create_tf_dev_feature(sample)

    def create_tf_dev_feature(self, sample):
        """
        Unlike the train feature, the dev set uses all passages. Both the input and the output are dicts with the shape
        [batch_size, num_passages, maxseqlen]
        """
        posdoc, negdoc, negdoc_id = sample["pos_bert_input"], sample["neg_bert_input"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = (
            sample["pos_mask"],
            sample["pos_seg"],
            sample["neg_mask"],
            sample["neg_seg"],
        )
        label = sample["label"]

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):  # if value ist tensor
                value = value.numpy()  # get value of tensor
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {
            "pos_bert_input": _bytes_feature(tf.io.serialize_tensor(posdoc)),
            "pos_mask": _bytes_feature(tf.io.serialize_tensor(posdoc_mask)),
            "pos_seg": _bytes_feature(tf.io.serialize_tensor(posdoc_seg)),
            "neg_bert_input": _bytes_feature(tf.io.serialize_tensor(negdoc)),
            "neg_mask": _bytes_feature(tf.io.serialize_tensor(negdoc_mask)),
            "neg_seg": _bytes_feature(tf.io.serialize_tensor(negdoc_seg)),
            "label": _bytes_feature(tf.io.serialize_tensor(label)),
        }

        return [feature]

    def parse_tf_train_example(self, example_proto):
        return self.parse_tf_dev_example(example_proto)

    def parse_tf_dev_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([self.config["numpassages"], self.config["maxseqlen"]])

            return parsed_tensor

        def parse_label_tensor(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.float32)
            parsed_tensor.set_shape([2])

            return parsed_tensor

        pos_bert_input = tf.map_fn(parse_tensor_as_int, parsed_example["pos_bert_input"], dtype=tf.int64)
        pos_mask = tf.map_fn(parse_tensor_as_int, parsed_example["pos_mask"], dtype=tf.int64)
        pos_seg = tf.map_fn(parse_tensor_as_int, parsed_example["pos_seg"], dtype=tf.int64)
        neg_bert_input = tf.map_fn(parse_tensor_as_int, parsed_example["neg_bert_input"], dtype=tf.int64)
        neg_mask = tf.map_fn(parse_tensor_as_int, parsed_example["neg_mask"], dtype=tf.int64)
        neg_seg = tf.map_fn(parse_tensor_as_int, parsed_example["neg_seg"], dtype=tf.int64)
        label = tf.map_fn(parse_label_tensor, parsed_example["label"], dtype=tf.float32)

        return (pos_bert_input, pos_mask, pos_seg, neg_bert_input, neg_mask, neg_seg), label

    def id2vec(self, qid, posid, negid=None, label=None):
        """
        See parent class for docstring
        """
        assert label is not None

        tokenizer = self.tokenizer
        maxseqlen = self.config["maxseqlen"]

        query_toks = self.qid2toks[qid]
        pos_bert_inputs = []
        pos_bert_masks = []
        pos_bert_segs = []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self.docid2passages[posid]
        for tokenized_passage in pos_passages:
            input_line = ["[CLS]"] + query_toks + ["[SEP]"] + tokenized_passage + ["[SEP]"]
            if len(input_line) > maxseqlen:
                input_line = input_line[:maxseqlen]
                input_line[-1] = "[SEP]"

            padded_input_line = padlist(input_line, padlen=self.config["maxseqlen"], pad_token=self.pad_tok)
            pos_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
            pos_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
            pos_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))

        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "qid": qid,
            "posdocid": posid,
            "pos_bert_input": np.array(pos_bert_inputs, dtype=np.long),
            "pos_mask": np.array(pos_bert_masks, dtype=np.long),
            "pos_seg": np.array(pos_bert_segs, dtype=np.long),
            "negdocid": "",
            "neg_bert_input": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "neg_mask": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "neg_seg": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "label": np.array(label, dtype=np.float32),
        }

        if negid:
            neg_bert_inputs = []
            neg_bert_masks = []
            neg_bert_segs = []
            neg_passages = self.docid2passages[negid]
            for tokenized_passage in neg_passages:
                input_line = ["[CLS]"] + query_toks + ["[SEP]"] + tokenized_passage + ["[SEP]"]
                if len(input_line) > maxseqlen:
                    input_line = input_line[:maxseqlen]
                    input_line[-1] = "[SEP]"

                padded_input_line = padlist(input_line, padlen=self.config["maxseqlen"], pad_token=self.pad_tok)
                neg_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
                neg_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
                neg_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))

            if not neg_bert_inputs:
                raise MissingDocError(qid, negid)

            data["negdocid"] = negid
            data["neg_bert_input"] = np.array(neg_bert_inputs, dtype=np.long)
            data["neg_mask"] = np.array(neg_bert_masks, dtype=np.long)
            data["neg_seg"] = np.array(neg_bert_segs, dtype=np.long)

        return data
