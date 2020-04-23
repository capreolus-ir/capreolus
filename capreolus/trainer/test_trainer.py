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
        return {"query": np.array([1, 2, 3, 4], dtype=np.long), "posdoc": np.array([1, 1, 1, 1], dtype=np.float), "negdoc": np.array([2, 2, 2, 2], dtype=np.long), 'qid': 1, 'posdocid': 'posdoc1', 'negdocid': 'negdoc1', 'query_idf': np.array([0.1, 0.1, 0.2, 0.1], dtype=np.float)}

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
            "usecache": False
        }
    )

    tf_record_filenames = trainer.convert_to_tf_record(train_dataset)
    for filename in tf_record_filenames:
        assert os.path.isfile(filename)

    tf_record_dataset = trainer.load_tf_records_from_file(tf_record_filenames)
    dataset = tf_record_dataset.batch(2)

    num_batches = 0
    for idx, batch in enumerate(dataset):
        tf.debugging.assert_equal(batch['qid'], tf.ones(2, dtype=tf.int64))
        tf.debugging.assert_equal(
            batch['query'],
            tf.convert_to_tensor(
                np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), dtype=tf.float32
            )
        )
        tf.debugging.assert_equal(
            batch['query_idf'],
            tf.convert_to_tensor(
                np.array([[0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.1]]), dtype=tf.float32
            )
        )
        tf.debugging.assert_equal(batch['posdoc_id'], ['posdoc1', 'posdoc1'])
        tf.debugging.assert_equal(
            batch['posdoc'],
            tf.convert_to_tensor(
                np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
                dtype=tf.float32
            )
        )
        tf.debugging.assert_equal(batch['negdoc_id'], ['negdoc1', 'negdoc1'])
        tf.debugging.assert_equal(
            batch['negdoc'],
            tf.convert_to_tensor(
                np.array([[2, 2, 2, 2], [2, 2, 2, 2]]),
                dtype=tf.float32
            )
        )
        num_batches = idx

    assert num_batches + 1 == 8 * 16


