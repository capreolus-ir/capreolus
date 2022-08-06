import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils
import tensorflow as tf

from capreolus import constants, get_logger

logger = get_logger(__name__)

embedding_paths = {
    "glove6b": "glove/light/glove.6B.300d",
    "glove6b.50d": "glove/light/glove.6B.50d",
    "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
    "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
}

pad_tok = "<pad>"


def load_pretrained_embeddings(embedding_name):
    if embedding_name not in embedding_paths:
        raise ValueError(f"embedding name '{embedding_name}' is not a recognized embedding: {sorted(embedding_paths.keys())}")

    embedding_cache = constants["CACHE_BASE_PATH"] / "embeddings"
    numpy_cache = embedding_cache / (embedding_name + ".npy")
    vocab_cache = embedding_cache / (embedding_name + ".vocab.txt")

    if numpy_cache.exists() and vocab_cache.exists():
        logger.debug("loading embeddings from %s", numpy_cache)
        stoi, itos = load_vocab_file(vocab_cache)
        embeddings = np.load(numpy_cache, mmap_mode="r").reshape(len(stoi), -1)

        return embeddings, itos, stoi

    logger.debug("preparing embeddings and vocab")
    magnitude = Magnitude(MagnitudeUtils.download_model(embedding_paths[embedding_name], download_dir=embedding_cache))

    terms, vectors = zip(*((term, vector) for term, vector in magnitude))
    pad_vector = np.zeros(magnitude.dim, dtype=np.float32)
    terms = [pad_tok] + list(terms)
    vectors = np.array([pad_vector] + list(vectors), dtype=np.float32)
    itos = {idx: term for idx, term in enumerate(terms)}

    logger.debug("saving embeddings to %s", numpy_cache)
    np.save(numpy_cache, vectors, allow_pickle=False)
    save_vocab_file(itos, vocab_cache)
    stoi = {s: i for i, s in itos.items()}

    return vectors, itos, stoi


def load_vocab_file(fn):
    stoi, itos = {}, {}
    with open(fn, "rt") as f:
        for idx, line in enumerate(f):
            term = line.strip()
            stoi[term] = idx
            itos[idx] = term

    assert itos[0] == pad_tok
    return stoi, itos


def save_vocab_file(itos, fn):
    with open(fn, "wt") as outf:
        for idx, term in sorted(itos.items()):
            print(term, file=outf)


class MultipleTrainingPassagesMixin:
    """
    Prepare and parse TF training feature that contain multiple passage per query.
    That is, the "pos_bert_input" features prepared by extractor's `id2vec()` function should have 3 dimension
    """

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


class SingleTrainingPassagesMixin:
    """
    Prepare and parse TF training feature that contain single passage per query.
    That is, the "pos_bert_input" features prepared by extractor's `id2vec()` function should have 2 dimension
    """

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
