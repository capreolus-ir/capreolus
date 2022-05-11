import tensorflow as tf
import numpy as np

from capreolus import get_logger
from capreolus.utils.exceptions import MissingDocError
from . import Extractor
from .bertpassage import BertPassage

logger = get_logger(__name__)


@Extractor.register
class LCEBertPassage(BertPassage):
    module_name = "LCEbertpassage"

    config_spec = BertPassage.config_spec

    def create_tf_train_feature(self, sample):
        """
        Returns a set of features from a doc.
        Of the num_passages passages that are present in a document, we use only a subset of it.
        params:
        sample - A dict where each entry has the shape [batch_size, num_passages, maxseqlen]
        Returns a list of features. Each feature is a dict, and each value in the dict has the shape [batch_size, maxseqlen].
        Yes, the output shape is different to the input shape because we sample from the passages.
        """
        num_passages = self.config["numpassages"]

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte. Our features are multi-dimensional tensors."""
            if isinstance(value, type(tf.constant(0))):  # if value ist tensor
                value = value.numpy()  # get value of tensor
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def transpose_neg_input(neg_inp):
            return tf.cast(tf.transpose(neg_inp, perm=[1, 0, 2]), tf.int64)

        posdoc, negdoc, negdoc_id = sample["pos_bert_input"], sample["neg_bert_input"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = (
            sample["pos_mask"],
            sample["pos_seg"],
            sample["neg_mask"],
            sample["neg_seg"],
        )
        label = sample["label"]
        features = []

        negdoc = transpose_neg_input(negdoc)
        negdoc_seg = transpose_neg_input(negdoc_seg)
        negdoc_mask = transpose_neg_input(negdoc_mask)

        for i in range(num_passages):
            if i > 0 and self.rng.random() > self.config["prob"]:
                continue

            bert_input_line = posdoc[i]
            bert_input_line = " ".join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(list(bert_input_line)))
            passage = bert_input_line.split(self.sep_tok)[-2]

            # Ignore empty passages as well
            if passage.strip() == self.pad_tok:
                continue

            feature = {
                "pos_bert_input": _bytes_feature(tf.io.serialize_tensor(posdoc[i])),
                "pos_mask": _bytes_feature(tf.io.serialize_tensor(posdoc_mask[i])),
                "pos_seg": _bytes_feature(tf.io.serialize_tensor(posdoc_seg[i])),
                "neg_bert_input": _bytes_feature(tf.io.serialize_tensor(negdoc[i])),
                "neg_mask": _bytes_feature(tf.io.serialize_tensor(negdoc_mask[i])),
                "neg_seg": _bytes_feature(tf.io.serialize_tensor(negdoc_seg[i])),
                "label": _bytes_feature(tf.io.serialize_tensor(label[i])),
            }
            features.append(feature)

        return features

    def parse_tf_train_example(self, example_proto):
        maxseqlen = self.config["maxseqlen"]

        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([maxseqlen])
            return parsed_tensor

        def parse_neg_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            return parsed_tensor

        def parse_label_tensor(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.float32)
            return parsed_tensor

        pos_bert_input = tf.map_fn(parse_tensor_as_int, parsed_example["pos_bert_input"], dtype=tf.int64)
        pos_mask = tf.map_fn(parse_tensor_as_int, parsed_example["pos_mask"], dtype=tf.int64)
        pos_seg = tf.map_fn(parse_tensor_as_int, parsed_example["pos_seg"], dtype=tf.int64)
        neg_bert_input = tf.map_fn(parse_neg_tensor_as_int, parsed_example["neg_bert_input"], dtype=tf.int64)
        neg_mask = tf.map_fn(parse_neg_tensor_as_int, parsed_example["neg_mask"], dtype=tf.int64)
        neg_seg = tf.map_fn(parse_neg_tensor_as_int, parsed_example["neg_seg"], dtype=tf.int64)
        label = tf.map_fn(parse_label_tensor, parsed_example["label"], dtype=tf.float32)

        return (pos_bert_input, pos_mask, pos_seg, neg_bert_input, neg_mask, neg_seg), label

    def id2vec(self, qid, posid, negids=None, label=None, **kwargs):
        """
        See parent class for docstring
        """
        training = kwargs.get("training", True)  # default to be training
        assert label is not None
        maxseqlen = self.config["maxseqlen"]
        numpassages = self.config["numpassages"]

        query_toks = self.qid2toks[qid]
        pos_bert_inputs, pos_bert_masks, pos_bert_segs = [], [], []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self._get_passages(posid)
        for tokenized_passage in pos_passages:
            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            pos_bert_inputs.append(inp)
            pos_bert_masks.append(mask)
            pos_bert_segs.append(seg)

        label = [label] if training else label
        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "qid": qid,
            "posdocid": posid,
            "pos_bert_input": np.array(pos_bert_inputs, dtype=np.long),
            "pos_mask": np.array(pos_bert_masks, dtype=np.long),
            "pos_seg": np.array(pos_bert_segs, dtype=np.long),
            "negdocid": "",
            "neg_bert_input": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "neg_mask": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "neg_seg": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "label": np.repeat(np.array(label, dtype=np.float32), numpassages, 0),
        }

        if negids is None:
            return data

        data["negdocid"] = []
        data["neg_bert_input"] = []
        data["neg_mask"] = []
        data["neg_seg"] = []
        for negid in negids:
            neg_bert_inputs, neg_bert_masks, neg_bert_segs = [], [], []
            neg_passages = self._get_passages(negid)
            for tokenized_passage in neg_passages:
                inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
                neg_bert_inputs.append(inp)
                neg_bert_masks.append(mask)
                neg_bert_segs.append(seg)

            if not neg_bert_inputs:
                raise MissingDocError(qid, negid)

            data["negdocid"].append(negid)
            data["neg_bert_input"].append(np.array(neg_bert_inputs, dtype=np.long))
            data["neg_mask"].append(np.array(neg_bert_masks, dtype=np.long))
            data["neg_seg"].append(np.array(neg_bert_segs, dtype=np.long))
        return data
