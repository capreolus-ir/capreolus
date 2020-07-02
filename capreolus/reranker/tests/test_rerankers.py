import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pymagnitude import Magnitude

from capreolus import Reranker, module_registry
from capreolus.benchmark import DummyBenchmark
from capreolus.extractor.deeptileextractor import DeepTileExtractor
from capreolus.extractor.embedtext import EmbedText
from capreolus.extractor.slowembedtext import SlowEmbedText
from capreolus.reranker.CDSSM import CDSSM
from capreolus.reranker.DeepTileBar import DeepTileBar
from capreolus.reranker.DSSM import DSSM
from capreolus.reranker.HINT import HINT
from capreolus.reranker.KNRM import KNRM
from capreolus.reranker.PACRR import PACRR
from capreolus.reranker.POSITDRMM import POSITDRMM
from capreolus.reranker.TFKNRM import TFKNRM
from capreolus.reranker.TK import TK
from capreolus.sampler import PredDataset, TrainDataset
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache

rerankers = set(module_registry.get_module_names("reranker"))


@pytest.mark.parametrize("reranker_name", rerankers)
def test_reranker_creatable(tmpdir_as_cache, dummy_index, reranker_name):
    provide = {"collection": dummy_index.collection, "index": dummy_index}
    reranker = Reranker.create(reranker_name, provide=provide)


def test_knrm_pytorch(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_load_embeddings(self):
        self.embeddings = np.zeros((1, 50))
        self.stoi = {"<pad>": 0}
        self.itos = {v: k for k, v in self.stoi.items()}

    monkeypatch.setattr(EmbedText, "_load_pretrained_embeddings", fake_load_embeddings)

    reranker = KNRM(
        {
            "gradkernels": True,
            "scoretanh": False,
            "singlefc": True,
            "finetune": False,
            "trainer": {"niters": 1, "itersize": 4, "batch": 2},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_knrm_tf(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = TFKNRM(
        {"gradkernels": True, "finetune": False, "trainer": {"niters": 1, "itersize": 4, "batch": 2}},
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best.index")


def test_pacrr(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_load_embeddings(self):
        self.embeddings = np.zeros((1, 50))
        self.stoi = {"<pad>": 0}
        self.itos = {v: k for k, v in self.stoi.items()}

    monkeypatch.setattr(EmbedText, "_load_pretrained_embeddings", fake_load_embeddings)

    reranker = PACRR(
        {
            "nfilters": 32,
            "idf": True,
            "kmax": 2,
            "combine": 32,
            "nonlinearity": "relu",
            "trainer": {"niters": 1, "itersize": 4, "batch": 2},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_dssm_unigram(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    reranker = DSSM({"nhiddens": "56", "trainer": {"niters": 1, "itersize": 4, "batch": 2}}, provide={"index": dummy_index})
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_tk(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = TK(
        {
            "gradkernels": True,
            "scoretanh": False,
            "singlefc": True,
            "projdim": 32,
            "ffdim": 100,
            "numlayers": 2,
            "numattheads": 4,
            "alpha": 0.5,
            "usemask": False,
            "usemixer": True,
            "finetune": True,
            "trainer": {"niters": 1, "itersize": 4, "batch": 2},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_tk_get_mask(tmpdir, dummy_index, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = TK(
        {
            "gradkernels": True,
            "scoretanh": False,
            "singlefc": True,
            "projdim": 16,
            "ffdim": 100,
            "numlayers": 1,
            "numattheads": 2,
            "alpha": 0.5,
            "usemask": True,
            "usemixer": True,
            "finetune": True,
            "trainer": {"niters": 1, "itersize": 4, "batch": 2},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    # 3 batches, each of seq len 4, and embedding dim 8
    embedding = torch.ones(3, 4, 8)
    # Set the 2nd and last token in first batch as pad
    embedding[0, 1] = torch.zeros(8)
    embedding[0, 3] = torch.zeros(8)

    # set the first and third token in second batch as pad
    embedding[1, 0] = torch.zeros(8)
    embedding[1, 2] = torch.zeros(8)

    mask = reranker.model.get_mask(embedding)

    assert torch.equal(
        mask[0],
        torch.tensor(
            [
                [0, float("-inf"), 0, float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [0, float("-inf"), 0, float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
            ],
            dtype=torch.float,
        ),
    )

    assert torch.equal(
        mask[1],
        torch.tensor(
            [
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [float("-inf"), 0, float("-inf"), 0],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [float("-inf"), 0, float("-inf"), 0],
            ],
            dtype=torch.float,
        ),
    )

    assert torch.equal(mask[2], torch.zeros(4, 4, dtype=torch.float))


def test_deeptilebars(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    reranker = DeepTileBar(
        {
            "name": "DeepTileBar",
            "passagelen": 30,
            "numberfilter": 3,
            "lstmhiddendim": 3,
            "linearhiddendim1": 32,
            "linearhiddendim2": 16,
            "trainer": {"niters": 1, "itersize": 4, "batch": 2},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_HINT(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = HINT(
        {"spatialGRU": 2, "LSTMdim": 6, "kmax": 10, "trainer": {"niters": 1, "itersize": 2, "batch": 1}},
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_POSITDRMM(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = POSITDRMM({"trainer": {"niters": 1, "itersize": 4, "batch": 2}}, provide={"index": dummy_index})
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()
    reranker.searcher_scores = {"301": {"LA010189-0001": 2, "LA010189-0002": 1}}
    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_CDSSM(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    reranker = CDSSM(
        {
            "nkernel": 3,
            "nfilter": 1,
            "nhiddens": 30,
            "windowsize": 3,
            "dropoutrate": 0,
            "trainer": {"niters": 1, "itersize": 2, "batch": 1},
        },
        provide={"index": dummy_index},
    )
    extractor = reranker.extractor
    metric = "map"
    benchmark = DummyBenchmark()

    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build_model()
    reranker.searcher_scores = {"301": {"LA010189-0001": 2, "LA010189-0002": 1}}
    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker.trainer.train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )
