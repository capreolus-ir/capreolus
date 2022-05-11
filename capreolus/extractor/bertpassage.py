import pickle
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


from capreolus.extractor import Extractor
from capreolus import Dependency, ConfigOption, get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError
from capreolus.tokenizer.punkt import PunktTokenizer

logger = get_logger(__name__)


@Extractor.register
class BertPassage(Extractor):
    """
    Extracts passages from the document to be later consumed by a BERT based model.
    Does NOT use all the passages. The first passages is always used. Use the `prob` config to control the probability
    of a passage being selected
    Gotcha: In Tensorflow the train tfrecords have shape (batch_size, maxseqlen) while dev tf records have the shape
    (batch_size, num_passages, maxseqlen). This is because during inference, we want to pool over the scores of the
    passages belonging to a doc
    """

    module_name = "bertpassage"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name=None),
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]

    config_spec = [
        ConfigOption("maxseqlen", 256, "Maximum input length (query+document)"),
        ConfigOption("maxqlen", 20, "Maximum query length"),
        ConfigOption("padq", False, "Always pad queries to maxqlen"),
        ConfigOption("usecache", False, "Should the extracted features be cached?"),
        ConfigOption("passagelen", 150, "Length of the extracted passage"),
        ConfigOption("stride", 100, "Stride"),
        ConfigOption("sentences", False, "Use a sentence tokenizer to form passages"),
        ConfigOption("numpassages", 16, "Number of passages per document"),
        ConfigOption(
            "prob",
            0.1,
            "The probability that a passage from the document will be used for training " "(the first passage is always used)",
        ),
    ]
    config_keys_not_in_path = ["usecache"]

    def build(self):
        self.pad = self.tokenizer.bert_tokenizer.pad_token_id
        self.cls = self.tokenizer.bert_tokenizer.cls_token_id
        self.sep = self.tokenizer.bert_tokenizer.sep_token_id

        self.pad_tok = self.tokenizer.bert_tokenizer.pad_token
        self.cls_tok = self.tokenizer.bert_tokenizer.cls_token
        self.sep_tok = self.tokenizer.bert_tokenizer.sep_token

    def load_state(self, qids, docids):
        cache_fn = self.get_state_cache_file_path(qids, docids)
        logger.debug("loading state from: %s", cache_fn)
        with open(cache_fn, "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks}
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "pos_bert_input": tf.io.FixedLenFeature([], tf.string),
            "pos_mask": tf.io.FixedLenFeature([], tf.string),
            "pos_seg": tf.io.FixedLenFeature([], tf.string),
            "neg_bert_input": tf.io.FixedLenFeature([], tf.string),
            "neg_mask": tf.io.FixedLenFeature([], tf.string),
            "neg_seg": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
        }

        return feature_description

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

        posdoc, negdoc, negdoc_id = sample["pos_bert_input"], sample["neg_bert_input"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = (
            sample["pos_mask"],
            sample["pos_seg"],
            sample["neg_mask"],
            sample["neg_seg"],
        )
        label = sample["label"]
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
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([self.config["maxseqlen"]])

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

    def parse_tf_dev_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([self.config["numpassages"], self.config["maxseqlen"]])

            return parsed_tensor

        def parse_label_tensor(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.float32)
            # parsed_tensor.set_shape([self.config["numpassages"], 2])
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
    
    def _filter_inputs(self, bert_inputs, bert_masks, bert_segs, n_valid_psg):
        """Preserve only one passage from all available passages."""
        assert n_valid_psg <= len(bert_inputs), f"Passages only have {len(bert_inputs)} entries, but got {n_valid_psg} valid passages."
        valid_indexes = list(range(0, n_valid_psg))
        if len(valid_indexes) == 0:
            valid_indexes = [0]
        random_i = self.rng.choice(valid_indexes)
        return list(map(lambda arr: arr[random_i], [bert_inputs, bert_masks, bert_segs]))
 
    def _encode_inputs(self, query_toks, passages):
        """Convert the query and passages into BERT inputs, mask, segments."""
        bert_inputs, bert_masks, bert_segs = [], [], [] 
        n_valid_psg = 0
        for tokenized_passage in passages:
            if tokenized_passage != [self.pad_tok]:  # end of the passage
                n_valid_psg += 1

            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            bert_inputs.append(inp)
            bert_masks.append(mask)
            bert_segs.append(seg)

        return bert_inputs, bert_masks, bert_segs, n_valid_psg

    def _get_passages(self, docid):
        doc = self.index.get_doc(docid)
        if not self.config["sentences"]:
            return self._get_sliding_window_passages(doc)
        else:
            return self._get_sent_passages(doc)

    def _get_sent_passages(self, doc):
        passages = []
        punkt = PunktTokenizer()
        numpassages = self.config["numpassages"]
        for sentence in punkt.tokenize(doc):
            if len(passages) >= numpassages:
                break
            passages.extend(self._chunk_sent(sentence, self.config["passagelen"]))

        if numpassages != 0:
            passages = passages[:numpassages]

        n_actual_passages = len(passages)
        for _ in range(numpassages - n_actual_passages):
            # randomly use one of previous passages when the document is exhausted
            # append empty passages
            passages.append([""])

        assert len(passages) == numpassages or len(numpassages) == 0
        return sorted(passages, key=len)

    def _get_sliding_window_passages(self, doc):
        """
        Extract passages from the doc.
        If there are too many passages, keep the first and the last one and sample from the rest.
        If there are not enough packages, pad.
        """
        passages = []
        numpassages = self.config["numpassages"]
        doc = self.tokenizer.tokenize(doc)

        for i in range(0, len(doc), self.config["stride"]):
            if i >= len(doc):
                assert len(passages) > 0, f"no passage can be built from empty document {doc}"
                break
            passages.append(doc[i : i + self.config["passagelen"]])

        n_actual_passages = len(passages)
        # If we have a more passages than required, keep the first and last, and sample from the rest
        if n_actual_passages > numpassages:
            if numpassages > 1:
                # passages = [passages[0]] + list(self.rng.choice(passages[1:-1], numpassages - 2, replace=False)) + [passages[-1]]
                passages = passages[:numpassages]
            else:
                passages = [passages[0]]
        else:
            # Pad until we have the required number of passages
            passages.extend([[self.pad_tok] for _ in range(numpassages - n_actual_passages)])

        assert len(passages) == numpassages
        return passages

    # from https://github.com/castorini/birch/blob/2dd0401ebb388a1c96f8f3357a064164a5db3f0e/src/utils/doc_utils.py#L73
    def _chunk_sent(self, sent, max_len):
        words = self.tokenizer.tokenize(sent)

        if len(words) <= max_len:
            return [words]

        chunked_sents = []
        size = int(len(words) / max_len)
        for i in range(0, size):
            seq = words[i * max_len : (i + 1) * max_len]
            chunked_sents.append(seq)
        return chunked_sents

    def _build_vocab(self, qids, docids, topics):
        """only build vocab for queries as the size of docidid2document would be large for some of the document retrieval collection."""
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            logger.info("Vocabulary loaded from cache")
            self.load_state(qids, docids)
        else:
            logger.info("Building BertPassage vocabulary")
            self.qid2toks = {qid: self.tokenizer.tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "qid2toks") and len(self.qid2toks)

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()
        self._build_vocab(qids, docids, topics)

    def _prepare_bert_input(self, query_toks, psg_toks):
        maxseqlen, maxqlen = self.config["maxseqlen"], self.config["maxqlen"]
        if len(query_toks) > maxqlen:
            logger.warning(f"Truncating query from {len(query_toks)} to {maxqlen}")
            query_toks = query_toks[:maxqlen]
        else:  # if the len(query_toks) <= maxqlen, whether to pad it
            if self.config["padq"]:
                query_toks = padlist(query_toks, padlen=maxqlen, pad_token=self.pad_tok)
        psg_toks = psg_toks[: maxseqlen - len(query_toks) - 3]

        psg_toks = " ".join(psg_toks).split()  # in case that psg_toks is np.array
        input_line = [self.cls_tok] + query_toks + [self.sep_tok] + psg_toks + [self.sep_tok]
        padded_input_line = padlist(input_line, padlen=maxseqlen, pad_token=self.pad_tok)
        inp = self.tokenizer.convert_tokens_to_ids(padded_input_line)
        mask = [0 if tok != self.pad_tok else 1 for tok in input_line] + [0] * (len(padded_input_line) - len(input_line))
        seg = [0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2)
        return inp, mask, seg
 
    def id2vec(self, qid, posid, negid=None, label=None, *args, **kwargs):
        """
        See parent class for docstring
        """
        training = kwargs.get("training", True) # default to be training

        assert label is not None
        maxseqlen = self.config["maxseqlen"]
        numpassages = self.config["numpassages"]

        query_toks = self.qid2toks[qid]

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self._get_passages(posid)
        pos_bert_inputs, pos_bert_masks, pos_bert_segs, n_valid_psg = self._encode_inputs(query_toks, pos_passages)
        if training:
            pos_bert_inputs, pos_bert_masks, pos_bert_segs = self._filter_inputs(pos_bert_inputs, pos_bert_masks, pos_bert_segs, n_valid_psg)
        else:
            # inp_shape, exp_shape = pos_bert_inputs.shape, (numpassages, maxseqlen)
            # assert inp_shape ==  exp_shape, f"Inference data should be have shape {exp_shape}, but got {inp_shape}."
            assert len(pos_bert_inputs) == numpassages

        pos_bert_inputs, pos_bert_masks, pos_bert_segs = map(
            lambda lst: np.array(lst, dtype=np.long), [pos_bert_inputs, pos_bert_masks, pos_bert_segs])

        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "qid": qid,
            "posdocid": posid,
            "pos_bert_input": pos_bert_inputs,
            "pos_mask": pos_bert_masks,
            "pos_seg": pos_bert_segs,
            "negdocid": "",
            "neg_bert_input": np.zeros_like(pos_bert_inputs, dtype=np.long),
            "neg_mask": np.zeros_like(pos_bert_masks, dtype=np.long),
            "neg_seg": np.zeros_like(pos_bert_segs, dtype=np.long),
            # "label": np.repeat(np.array([label], dtype=np.float32), numpassages, 0), 
            "label": np.array(label, dtype=np.float32), 
            # ^^^ not change the shape of the label as it is only needed during training
        }

        if not negid:
            return data

        neg_passages = self._get_passages(negid)
        neg_bert_inputs, neg_bert_masks, neg_bert_segs, n_valid_psg = self._encode_inputs(query_toks, neg_passages)
        if training:
            neg_bert_inputs, neg_bert_masks, neg_bert_segs = self._filter_inputs(neg_bert_inputs, neg_bert_masks, neg_bert_segs, n_valid_psg)
        else:
            # inp_shape, exp_shape = neg_bert_inputs.shape, (numpassages, maxseqlen)
            # assert inp_shape ==  exp_shape, f"Inference data should be have shape {exp_shape}, but got {inp_shape}."
            assert len(neg_bert_inputs) == numpassages

        if not neg_bert_inputs:
            raise MissingDocError(qid, negid)

        data["negdocid"] = negid
        data["neg_bert_input"] = np.array(neg_bert_inputs, dtype=np.long)
        data["neg_mask"] = np.array(neg_bert_masks, dtype=np.long)
        data["neg_seg"] = np.array(neg_bert_segs, dtype=np.long)
        return data
