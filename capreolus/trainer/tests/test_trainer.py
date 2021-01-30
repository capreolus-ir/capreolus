import collections
import os

import numpy as np
import tensorflow as tf

from capreolus.benchmark import DummyBenchmark
from capreolus.sampler import TrainTripletSampler
from capreolus.trainer.tensorflow import TensorflowTrainer
from capreolus.extractor.slowembedtext import SlowEmbedText
from capreolus.reranker.TFKNRM import TFKNRM
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache


def test_tf_get_tf_dataset(monkeypatch):
    benchmark = DummyBenchmark()
    extractor = SlowEmbedText(
        {"maxdoclen": 4, "maxqlen": 4, "tokenizer": {"keepstops": True}},
        provide={"collection": benchmark.collection, "benchmark": benchmark},
    )
    training_judgments = benchmark.qrels.copy()
    train_dataset = TrainTripletSampler()
    train_dataset.prepare(training_judgments, training_judgments, extractor)

    reranker = collections.namedtuple("reranker", "extractor")(extractor=extractor)

    def mock_id2vec(*args, **kwargs):
        return {
            "query": np.array([1, 2, 3, 4], dtype=np.long),
            "posdoc": np.array([1, 1, 1, 1], dtype=np.long),
            "negdoc": np.array([2, 2, 2, 2], dtype=np.long),
            "qid": "1",
            "posdocid": "posdoc1",
            "negdocid": "negdoc1",
            "query_idf": np.array([0.1, 0.1, 0.2, 0.1], dtype=np.float),
        }

    monkeypatch.setattr(SlowEmbedText, "id2vec", mock_id2vec)
    trainer = TensorflowTrainer(
        {
            "name": "tensorflow",
            "batch": 2,
            "niters": 2,
            "itersize": 16,
            "lr": 0.001,
            "validatefreq": 1,
            "usecache": False,
            "tpuname": None,
            "tpuzone": None,
            "storage": None,
        }
    )

    tf_record_filenames = trainer.convert_to_tf_train_record(reranker, train_dataset)
    for filename in tf_record_filenames:
        assert os.path.isfile(filename)

    tf_record_dataset = trainer.load_tf_train_records_from_file(reranker, tf_record_filenames, 2)
    dataset = tf_record_dataset

    for idx, data_and_label in enumerate(dataset):
        batch, _ = data_and_label
        tf.debugging.assert_equal(batch[0], tf.convert_to_tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), dtype=tf.int64))
        tf.debugging.assert_equal(batch[1], tf.convert_to_tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]), dtype=tf.int64))
        tf.debugging.assert_equal(batch[2], tf.convert_to_tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), dtype=tf.int64))
        tf.debugging.assert_equal(
            batch[3], tf.convert_to_tensor(np.array([[0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.1]]), dtype=tf.float32)
        )


def test_tf_find_cached_tf_records(monkeypatch, dummy_index):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    benchmark = DummyBenchmark()
    reranker = TFKNRM(
        {"gradkernels": True, "finetune": False, "trainer": {"niters": 1, "itersize": 8, "batch": 2}},
        provide={"index": dummy_index, "benchmark": benchmark},
    )
    extractor = reranker.extractor

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainTripletSampler()
    train_dataset.prepare(train_run, benchmark.qrels, extractor)

    required_samples = 8
    reranker.trainer.convert_to_tf_train_record(reranker, train_dataset)
    assert reranker.trainer.find_cached_tf_records(train_dataset, required_samples) is not None
    assert reranker.trainer.find_cached_tf_records(train_dataset, required_samples - 4) is not None
    assert reranker.trainer.find_cached_tf_records(train_dataset, 24) is None

    reranker = TFKNRM(
        {"gradkernels": True, "finetune": False, "trainer": {"niters": 1, "itersize": 24, "batch": 6}},
        provide={"index": dummy_index},
    )
    reranker.extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    train_dataset.prepare(train_run, benchmark.qrels, extractor)
    reranker.trainer.convert_to_tf_train_record(reranker, train_dataset)
    assert reranker.trainer.find_cached_tf_records(train_dataset, 24) is not None
    assert reranker.trainer.find_cached_tf_records(train_dataset, 18) is not None
