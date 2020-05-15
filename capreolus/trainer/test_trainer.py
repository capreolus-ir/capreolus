import numpy as np
import os
import tensorflow as tf
from capreolus.benchmark import DummyBenchmark
from capreolus.extractor import EmbedText
from capreolus.sampler import TrainDataset
from capreolus.trainer import TensorFlowTrainer


def test_tf_get_tf_dataset(monkeypatch):
    benchmark = DummyBenchmark({"fold": "s1", "rundocsonly": True})
    extractor = EmbedText({"keepstops": True, "maxdoclen": 4, "maxqlen": 4})
    training_judgments = benchmark.qrels.copy()
    train_dataset = TrainDataset(training_judgments, training_judgments, extractor)

    reranker = {"extractor": extractor}

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

    monkeypatch.setattr(EmbedText, "id2vec", mock_id2vec)
    trainer = TensorFlowTrainer(
        {
            "_name": "tensorflow",
            "batch": 2,
            "niters": 2,
            "itersize": 16,
            "gradacc": 1,
            "lr": 0.001,
            "softmaxloss": True,
            "interactive": False,
            "fastforward": True,
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

    tf_record_dataset = trainer.load_tf_records_from_file(reranker, tf_record_filenames, 2)
    dataset = tf_record_dataset

    for idx, data_and_label in enumerate(dataset):
        batch, _ = data_and_label
        tf.debugging.assert_equal(batch[0], tf.convert_to_tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), dtype=tf.int64))
        tf.debugging.assert_equal(batch[1], tf.convert_to_tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]), dtype=tf.int64))
        tf.debugging.assert_equal(batch[2], tf.convert_to_tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), dtype=tf.int64))
        tf.debugging.assert_equal(
            batch[3], tf.convert_to_tensor(np.array([[0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.1]]), dtype=tf.float32)
        )
