from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.collection import DummyCollection
from capreolus.index import AnseriniIndex
from capreolus.tokenizer import AnseriniTokenizer
from capreolus.benchmark import DummyBenchmark
from capreolus.extractor import EmbedText

from capreolus.utils.exceptions import MissingDocError

MAXQLEN = 8
MAXDOCLEN = 7


def test_embedtext_creation(monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    extractor_cfg = {
        "_name": "embedtext",
        "index": "anserini",
        "tokenizer": "anserini",
        "embeddings": "glove6b",
        "zerounk": True,
        "calcidf": True,
        "maxqlen": MAXQLEN,
        "maxdoclen": MAXDOCLEN,
    }
    extractor = EmbedText(extractor_cfg)

    benchmark = DummyBenchmark({"_fold": "s1", "rundocsonly": False})
    collection = DummyCollection({"_name": "dummy"})

    index_cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(index_cfg)
    index.modules["collection"] = collection

    tok_cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor.modules["index"] = index
    extractor.modules["tokenizer"] = tokenizer

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())
    extractor.create(qids, docids, benchmark.topics[benchmark.query_type])
    expected_vocabs = [
        "lessdummy",
        "dummy",
        "doc",
        "hello",
        "greetings",
        "world",
        "from",
        "outer",
        "space",
        "<pad>",
    ]
    expected_stoi = {s: i for i, s in enumerate(expected_vocabs)}

    assert set(extractor.stoi.keys()) == set(expected_stoi.keys())

    assert extractor.embeddings.shape == (len(expected_vocabs), 8)
    for i in range(extractor.embeddings.shape[0]):
        if i == extractor.pad:
            assert extractor.embeddings[i].sum() < 1e-5
            continue

    return extractor


def test_embedtext_id2vec(monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    extractor_cfg = {
        "_name": "embedtext",
        "index": "anserini",
        "tokenizer": "anserini",
        "embeddings": "glove6b",
        "zerounk": True,
        "calcidf": True,
        "maxqlen": MAXQLEN,
        "maxdoclen": MAXDOCLEN,
    }
    extractor = EmbedText(extractor_cfg)

    benchmark = DummyBenchmark({"_fold": "s1", "rundocsonly": False})
    collection = DummyCollection({"_name": "dummy"})

    index_cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(index_cfg)
    index.modules["collection"] = collection

    tok_cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)

    extractor.modules["index"] = index
    extractor.modules["tokenizer"] = tokenizer

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())

    extractor.create(qids, docids, benchmark.topics[benchmark.query_type])

    docid1, docid2 = docids[0], docids[1]
    data = extractor.id2vec(qid, docid1, docid2)
    q, d1, d2, idf = [data[k] for k in ["query", "posdoc", "negdoc", "idfs"]]

    assert q.shape[0] == idf.shape[0]

    topics = benchmark.topics[benchmark.query_type]
    # emb_path = "glove/light/glove.6B.300d"
    # fullemb = Magnitude(MagnitudeUtils.download_model(emb_path))

    assert len(q) == MAXQLEN
    assert len(d1) == MAXDOCLEN
    assert len(d2) == MAXDOCLEN

    assert len([w for w in q if w.sum() != 0]) == len(
        topics[qid].strip().split()[:MAXQLEN]
    )
    assert len([w for w in d1 if w.sum() != 0]) == len(
        extractor["index"].get_doc(docid1).strip().split()[:MAXDOCLEN]
    )
    assert len([w for w in d2 if w.sum() != 0]) == len(
        extractor["index"].get_doc(docid2).strip().split()[:MAXDOCLEN]
    )

    # check MissDocError
    error_thrown = False
    try:
        extractor.id2vec(qid, "0000000", "111111")
    except MissingDocError as err:
        error_thrown = True
        assert err.related_qid == qid
        assert err.missed_docid == "0000000"

    assert error_thrown


def fake_sampler():
    pass
