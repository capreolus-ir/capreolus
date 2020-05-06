import numpy as np
import os
import tensorflow as tf
from benchmark import DummyBenchmark
from extractor import EmbedText
from sampler import TrainDataset
from trainer import TensorFlowTrainer


def test_tf_get_tf_dataset(monkeypatch):
    benchmark = DummyBenchmark({"fold": "s1", "rundocsonly": True})
    extractor = EmbedText({"keepstops": True})
    training_judgments = benchmark.qrels.copy()
    train_dataset = TrainDataset(training_judgments, training_judgments, extractor)

    def mock_id2vec(*args, **kwargs):
        return {
            "query": np.array([1, 2, 3, 4], dtype=np.long),
            "posdoc": np.array([1, 1, 1, 1], dtype=np.float),
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
            "maxdoclen": 4,
            "maxqlen": 4,
            "batch": 8,
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
            "gcsbucket": None,
        }
    )

    tf_record_filenames = trainer.convert_to_tf_train_record(train_dataset)
    for filename in tf_record_filenames:
        assert os.path.isfile(filename)

    tf_record_dataset = trainer.load_tf_records_from_file(tf_record_filenames)
    dataset = tf_record_dataset.batch(2)

    for idx, data_and_label in enumerate(dataset):
        batch, _ = data_and_label
        tf.debugging.assert_equal(batch[0], tf.convert_to_tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), dtype=tf.float32))
        tf.debugging.assert_equal(batch[1], tf.convert_to_tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]), dtype=tf.float32))
        tf.debugging.assert_equal(batch[2], tf.convert_to_tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), dtype=tf.float32))
        tf.debugging.assert_equal(
            batch[3], tf.convert_to_tensor(np.array([[0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.1]]), dtype=tf.float32)
        )
