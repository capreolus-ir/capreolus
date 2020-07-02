import numpy as np
import torch
import torch.utils.data

from capreolus.benchmark import DummyBenchmark
from capreolus.extractor.embedtext import EmbedText
from capreolus.sampler import PredDataset, TrainDataset
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache


def test_train_sampler(monkeypatch, tmpdir):
    benchmark = DummyBenchmark()
    extractor = EmbedText({"tokenizer": {"keepstops": True}}, provide={"collection": benchmark.collection})
    training_judgments = benchmark.qrels.copy()
    train_dataset = TrainDataset(training_judgments, training_judgments, extractor)

    def mock_id2vec(*args, **kwargs):
        return {"query": np.array([1, 2, 3, 4]), "posdoc": np.array([1, 1, 1, 1]), "negdoc": np.array([2, 2, 2, 2])}

    monkeypatch.setattr(EmbedText, "id2vec", mock_id2vec)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for idx, batch in enumerate(dataloader):
        assert len(batch["query"]) == 32
        assert len(batch["posdoc"]) == 32
        assert len(batch["negdoc"]) == 32
        assert np.array_equal(batch["query"][0], np.array([1, 2, 3, 4]))
        assert np.array_equal(batch["query"][30], np.array([1, 2, 3, 4]))
        assert np.array_equal(batch["posdoc"][0], np.array([1, 1, 1, 1]))
        assert np.array_equal(batch["posdoc"][30], np.array([1, 1, 1, 1]))
        assert np.array_equal(batch["negdoc"][0], np.array([2, 2, 2, 2]))
        assert np.array_equal(batch["negdoc"][30], np.array([2, 2, 2, 2]))

        # Just making sure that the dataloader can do multiple iterations
        if idx > 3:
            break


def test_pred_sampler(monkeypatch, tmpdir):
    benchmark = DummyBenchmark()
    extractor = EmbedText({"tokenizer": {"keepstops": True}}, provide={"collection": benchmark.collection})
    search_run = {"301": {"LA010189-0001": 50, "LA010189-0002": 100}}
    pred_dataset = PredDataset(search_run, extractor)

    def mock_id2vec(*args, **kwargs):
        return {"query": np.array([1, 2, 3, 4]), "posdoc": np.array([1, 1, 1, 1])}

    monkeypatch.setattr(EmbedText, "id2vec", mock_id2vec)
    dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=2)
    for idx, batch in enumerate(dataloader):
        print(idx, batch)
        assert len(batch["query"]) == 2
        assert len(batch["posdoc"]) == 2
        assert batch.get("negdoc") is None
        assert np.array_equal(batch["query"][0], np.array([1, 2, 3, 4]))
        assert np.array_equal(batch["query"][1], np.array([1, 2, 3, 4]))
        assert np.array_equal(batch["posdoc"][0], np.array([1, 1, 1, 1]))
        assert np.array_equal(batch["posdoc"][1], np.array([1, 1, 1, 1]))
