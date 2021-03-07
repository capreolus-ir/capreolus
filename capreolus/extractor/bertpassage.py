import pickle
import ir_datasets
import torch
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
from transformers import BertTokenizerFast
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
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]

    config_spec = [
        ConfigOption("maxseqlen", 256, "Maximum input length (query+document)"),
        ConfigOption("maxqlen", 20, "Maximum query length"),
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

    def build(self):
        self.pad = self.tokenizer.bert_tokenizer.pad_token_id
        self.cls = self.tokenizer.bert_tokenizer.cls_token_id
        self.sep = self.tokenizer.bert_tokenizer.sep_token_id

        self.pad_tok = self.tokenizer.bert_tokenizer.pad_token
        self.cls_tok = self.tokenizer.bert_tokenizer.cls_token
        self.sep_tok = self.tokenizer.bert_tokenizer.sep_token

        if self.index.collection.module_name == "robust04":
            self.docs_store = ir_datasets.load("trec-robust04").docs_store()

    def get_doc(self, doc_id):
        if hasattr(self, "docs_store"):
            return self.docs_store.get(doc_id).text

        return self.index.get_doc(doc_id)

    def load_state(self, qids, docids):
        cache_fn = self.get_state_cache_file_path(qids, docids)
        logger.debug("loading state from: %s", cache_fn)
        with open(cache_fn, "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2passages = state_dict["docid2passages"]
            self.docid_to_passage_begin_token_obj = state_dict["docid_to_passage_begin_token_obj"]
            self.docid_to_doc_offsets_obj = state_dict["docid_to_doc_offset_obj"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {
                "qid2toks": self.qid2toks, "docid2passages": self.docid2passages, "docid_to_passage_begin_token_obj": self.docid_to_passage_begin_token_obj,
                "docid_to_doc_offset_obj": self.docid_to_doc_offsets_obj
            }
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
        features = []

        for i in range(num_passages):
            # Always use the first passage, then sample from the remaining passages
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
            parsed_tensor.set_shape([self.config["numpassages"], 2])

            return parsed_tensor

        pos_bert_input = tf.map_fn(parse_tensor_as_int, parsed_example["pos_bert_input"], dtype=tf.int64)
        pos_mask = tf.map_fn(parse_tensor_as_int, parsed_example["pos_mask"], dtype=tf.int64)
        pos_seg = tf.map_fn(parse_tensor_as_int, parsed_example["pos_seg"], dtype=tf.int64)
        neg_bert_input = tf.map_fn(parse_tensor_as_int, parsed_example["neg_bert_input"], dtype=tf.int64)
        neg_mask = tf.map_fn(parse_tensor_as_int, parsed_example["neg_mask"], dtype=tf.int64)
        neg_seg = tf.map_fn(parse_tensor_as_int, parsed_example["neg_seg"], dtype=tf.int64)
        label = tf.map_fn(parse_label_tensor, parsed_example["label"], dtype=tf.float32)

        return (pos_bert_input, pos_mask, pos_seg, neg_bert_input, neg_mask, neg_seg), label

    def _prepare_doc_psgs(self, doc):
        """
        Extract passages from the doc.
        If there are too many passages, keep the first and the last one and sample from the rest.
        If there are not enough packages, pad.
        """
        passages = []
        numpassages = self.config["numpassages"]
        tokenizer = self.tokenizer.bert_tokenizer.backend_tokenizer
        encoded_doc = tokenizer.encode(doc)
        doc = encoded_doc.tokens[1:-1]

        # For each wordpiece token, the beginning and end of the characters in the original doc
        doc_offsets = encoded_doc.offsets[1:-1]
        # doc = self.tokenizer.tokenize(doc)

        # To get the word in the doc corresponding to a word in the passage:
        # doc_offsets[passage_to_offsets[passage_id] + position of word in passage]
        passage_to_begin_token = {}

        for i in range(0, len(doc), self.config["stride"]):
            if i >= len(doc):
                assert len(passages) > 0, f"no passage can be built from empty document {doc}"
                break

            # Store the offset of the first character in the passage
            passage_to_begin_token[len(passages)] = i

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

        assert len(passages) == self.config["numpassages"]

        return passages, passage_to_begin_token, doc_offsets

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

    def _build_passages_from_sentences(self, docids):
        punkt = PunktTokenizer()

        for docid in tqdm(docids, "extract passages"):
            passages = []
            numpassages = self.config["numpassages"]
            for sentence in punkt.tokenize(self.get_doc(docid)):
                if len(passages) >= numpassages:
                    break

                passages.extend(self._chunk_sent(sentence, self.config["passagelen"]))

            if numpassages != 0:
                passages = passages[:numpassages]

                n_actual_passages = len(passages)
                for _ in range(numpassages - n_actual_passages):
                    # randomly use one of previous passages when the document is exhausted
                    # idx = random.randint(0, n_actual_passages - 1)
                    # passages.append(passages[idx])

                    # append empty passages
                    passages.append([""])

                assert len(passages) == self.config["numpassages"]

            self.docid2passages[docid] = sorted(passages, key=len)

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        elif self.config["sentences"]:
            self.docid2passages = {}
            self._build_passages_from_sentences(docids)
            self.qid2toks = {qid: self.tokenizer.tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.cache_state(qids, docids)
        else:
            logger.info("Building bertpassage vocabulary")

            self.qid2toks = {qid: self.tokenizer.tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}
            self.docid2passages = {}

            # Keeps track of the token index (relative to the original tokenized doc) where each passage begins
            self.docid_to_passage_begin_token_obj = {}
            # Maps each token in the tokenized doc to character ranges in the original doc
            self.docid_to_doc_offsets_obj = {}

            # To get the character range corresponding to a token in a passage:
            # self.doci_id_to_doc_offsets_obj[doc_id][self.doc_id_to_passage_begin_token_obj[docid][passage_id] + position of word in passage]
            # Yes, it's that verbose.

            for docid in tqdm(sorted(docids), desc="extract passages"):
                passages, passage_to_begin_token, doc_offsets = self._prepare_doc_psgs(self.get_doc(docid))
                self.docid2passages[docid] = passages
                self.docid_to_passage_begin_token_obj[docid] = passage_to_begin_token
                self.docid_to_doc_offsets_obj[docid] = doc_offsets

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "docid2passages") and len(self.docid2passages)

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()
        self.qid2toks = defaultdict(list)
        self.docid2passages = None

        self._build_vocab(qids, docids, topics)

    def _prepare_bert_input(self, query_toks, psg_toks):
        maxseqlen, maxqlen = self.config["maxseqlen"], self.config["maxqlen"]
        if len(query_toks) > maxqlen:
            query_toks = query_toks[:maxqlen]
            logger.warning(f"Truncating query from {len(query_toks)} to {maxqlen}")
        psg_toks = psg_toks[: maxseqlen - len(query_toks) - 3]

        psg_toks = " ".join(psg_toks).split()  # in case that psg_toks is np.array
        input_line = [self.cls_tok] + query_toks + [self.sep_tok] + psg_toks + [self.sep_tok]
        padded_input_line = padlist(input_line, padlen=maxseqlen, pad_token=self.pad_tok)
        inp = self.tokenizer.convert_tokens_to_ids(padded_input_line)
        mask = [1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line))
        seg = [0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2)
        return inp, mask, seg

    def get_diffir_weights_from_maxp(self, docid, *args):
        pass

    def get_diffir_weights_from_simmat(self, docid, simmat, passage_doc_mask):
        # assert simmat.shape == (self.config["numpassages"], self.config["maxqlen"], -1), "simmat shape is {}".format(simmat.shape)
        weights = []
        doc_offsets = self.docid_to_doc_offsets_obj[docid]
        for passage_id in range(self.config["numpassages"]):
            # Check for passages that are just padding
            if passage_id not in self.docid_to_passage_begin_token_obj[docid]:
                continue

            passage_begin_token_idx = self.docid_to_passage_begin_token_obj[docid][passage_id]
            num_doc_terms = simmat.shape[2]

            for doc_term_idx in range(num_doc_terms):
                # Avoid masked doc terms
                if passage_doc_mask[0][passage_id][0][doc_term_idx] == 0:
                    continue
                # Get the entire column - i.e we get all weights corresponding to each query term for a particular doc term
                doc_term_weights = simmat[passage_id][:, doc_term_idx]
                max_term_weight = torch.max(doc_term_weights, 0)[0].item()

                # Why? The [SEP] token that appears at the end will have a term weight, and won't be masked
                # However, we won't be able to map to the original doc. So, skip it
                # TODO: This could be potentially hiding a bug. I _think_ that I'm skipping the [SEP] token, but I could
                # be skipping something legit.
                if (passage_begin_token_idx + doc_term_idx) >= len(doc_offsets):
                    continue

                try:
                    char_range_in_original_doc = doc_offsets[passage_begin_token_idx + doc_term_idx]
                except IndexError:
                    logger.error("The mask is {}".format(passage_doc_mask[0][passage_id][0][doc_term_idx]))
                    logger.error("Max term weight was: {}".format(max_term_weight))
                    logger.error("passage_id: {}, passage_begin_token_idx: {}".format(passage_id, passage_begin_token_idx))
                    logger.error("doc_term_idx: {}".format(doc_term_idx))
                    logger.error("Doc position of term: {}".format(passage_begin_token_idx + doc_term_idx))
                    logger.error("Total number of tokens in original doc (i.e doc_offsets): {}".format(len(doc_offsets)))
                    raise

                weights.append([char_range_in_original_doc[0], char_range_in_original_doc[1], max_term_weight])

        return weights

    def id2vec(self, qid, posid, negid=None, label=None):
        """
        See parent class for docstring
        """
        assert label is not None
        maxseqlen = self.config["maxseqlen"]
        numpassages = self.config["numpassages"]

        query_toks = self.qid2toks[qid]
        pos_bert_inputs, pos_bert_masks, pos_bert_segs = [], [], []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self.docid2passages[posid]
        for tokenized_passage in pos_passages:
            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            pos_bert_inputs.append(inp)
            pos_bert_masks.append(mask)
            pos_bert_segs.append(seg)

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
            "label": np.repeat(np.array([label], dtype=np.float32), numpassages, 0),
        }

        if not negid:
            return data

        neg_bert_inputs, neg_bert_masks, neg_bert_segs = [], [], []
        neg_passages = self.docid2passages[negid]

        for tokenized_passage in neg_passages:
            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            neg_bert_inputs.append(inp)
            neg_bert_masks.append(mask)
            neg_bert_segs.append(seg)

        if not neg_bert_inputs:
            raise MissingDocError(qid, negid)

        data["negdocid"] = negid
        data["neg_bert_input"] = np.array(neg_bert_inputs, dtype=np.long)
        data["neg_mask"] = np.array(neg_bert_masks, dtype=np.long)
        data["neg_seg"] = np.array(neg_bert_segs, dtype=np.long)
        return data
