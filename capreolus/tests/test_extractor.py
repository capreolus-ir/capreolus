from collections import defaultdict

import nltk
import numpy as np
import pytest
from nltk import TextTilingTokenizer
from pymagnitude import Magnitude

from capreolus import Extractor, module_registry
from capreolus.benchmark import DummyBenchmark
from capreolus.collection import DummyCollection
from capreolus.extractor.bagofwords import BagOfWords
from capreolus.extractor.deeptileextractor import DeepTileExtractor
from capreolus.extractor.embedtext import EmbedText
from capreolus.extractor.slowembedtext import SlowEmbedText
from capreolus.index import AnseriniIndex
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache
from capreolus.tokenizer import AnseriniTokenizer
from capreolus.utils.exceptions import MissingDocError

MAXQLEN = 8
MAXDOCLEN = 7

extractors = set(module_registry.get_module_names("extractor"))


@pytest.mark.parametrize("extractor_name", extractors)
def test_extractor_creatable(tmpdir_as_cache, dummy_index, extractor_name):
    provide = {"index": dummy_index, "collection": dummy_index.collection}
    extractor = Extractor.create(extractor_name, provide=provide)


def test_embedtext_id2vec(monkeypatch):
    def fake_load_embeddings(self):
        vocab = ["<pad>", "lessdummy", "dummy", "doc", "hello", "greetings", "world", "from", "outer", "space"]
        self.embeddings = np.random.random((len(vocab), 50))
        self.embeddings[0, :] = 0
        self.stoi = {term: idx for idx, term in enumerate(vocab)}
        self.itos = {v: k for k, v in self.stoi.items()}

    monkeypatch.setattr(EmbedText, "_load_pretrained_embeddings", fake_load_embeddings)

    extractor_cfg = {"name": "embedtext", "embeddings": "glove6b", "calcidf": True, "maxqlen": MAXQLEN, "maxdoclen": MAXDOCLEN}
    extractor = EmbedText(extractor_cfg, provide={"collection": DummyCollection()})
    benchmark = DummyBenchmark()

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())

    extractor.preprocess(qids, docids, benchmark.topics[benchmark.query_type])

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

    assert len([w for w in q if w.sum() != 0]) == len(topics[qid].strip().split()[:MAXQLEN])
    assert len([w for w in d1 if w.sum() != 0]) == len(extractor.index.get_doc(docid1).strip().split()[:MAXDOCLEN])
    assert len([w for w in d2 if w.sum() != 0]) == len(extractor.index.get_doc(docid2).strip().split()[:MAXDOCLEN])

    # check MissDocError
    error_thrown = False
    try:
        extractor.id2vec(qid, "0000000", "111111")
    except MissingDocError as err:
        error_thrown = True
        assert err.related_qid == qid
        assert err.missed_docid == "0000000"

    assert error_thrown


def test_slowembedtext_creation(monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    index_cfg = {"name": "anserini", "indexstops": False, "stemmer": "porter", "collection": {"name": "dummy"}}
    index = AnseriniIndex(index_cfg)

    extractor_cfg = {
        "name": "slowembedtext",
        "embeddings": "glove6b",
        "zerounk": True,
        "calcidf": True,
        "maxqlen": MAXQLEN,
        "maxdoclen": MAXDOCLEN,
        "usecache": False,
    }
    extractor = SlowEmbedText(extractor_cfg, provide={"index": index})
    benchmark = DummyBenchmark()

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())
    extractor.preprocess(qids, docids, benchmark.topics[benchmark.query_type])
    expected_vocabs = ["lessdummy", "dummy", "doc", "hello", "greetings", "world", "from", "outer", "space", "<pad>"]
    expected_stoi = {s: i for i, s in enumerate(expected_vocabs)}

    assert set(extractor.stoi.keys()) == set(expected_stoi.keys())

    assert extractor.embeddings.shape == (len(expected_vocabs), 8)
    for i in range(extractor.embeddings.shape[0]):
        if i == extractor.pad:
            assert extractor.embeddings[i].sum() < 1e-5
            continue

    return extractor


def test_slowembedtext_id2vec(monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    extractor_cfg = {
        "name": "slowembedtext",
        "embeddings": "glove6b",
        "zerounk": True,
        "calcidf": True,
        "maxqlen": MAXQLEN,
        "maxdoclen": MAXDOCLEN,
        "usecache": False,
    }
    extractor = SlowEmbedText(extractor_cfg, provide={"collection": DummyCollection()})
    benchmark = DummyBenchmark()

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())

    extractor.preprocess(qids, docids, benchmark.topics[benchmark.query_type])

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

    assert len([w for w in q if w.sum() != 0]) == len(topics[qid].strip().split()[:MAXQLEN])
    assert len([w for w in d1 if w.sum() != 0]) == len(extractor.index.get_doc(docid1).strip().split()[:MAXDOCLEN])
    assert len([w for w in d2 if w.sum() != 0]) == len(extractor.index.get_doc(docid2).strip().split()[:MAXDOCLEN])

    # check MissDocError
    error_thrown = False
    try:
        extractor.id2vec(qid, "0000000", "111111")
    except MissingDocError as err:
        error_thrown = True
        assert err.related_qid == qid
        assert err.missed_docid == "0000000"

    assert error_thrown


def test_slowembedtext_caching(dummy_index, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return np.zeros((1, 8), dtype=np.float32), {0: "<pad>"}, {"<pad>": 0}

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    extractor_cfg = {
        "name": "slowembedtext",
        "embeddings": "glove6b",
        "zerounk": True,
        "calcidf": True,
        "maxqlen": MAXQLEN,
        "maxdoclen": MAXDOCLEN,
        "usecache": True,
    }
    extractor = SlowEmbedText(extractor_cfg, provide={"index": dummy_index})
    benchmark = DummyBenchmark()

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())

    assert not extractor.is_state_cached(qids, docids)

    extractor.preprocess(qids, docids, benchmark.topics[benchmark.query_type])

    assert extractor.is_state_cached(qids, docids)

    new_extractor = SlowEmbedText(extractor_cfg, provide={"index": dummy_index})

    assert new_extractor.is_state_cached(qids, docids)
    new_extractor._build_vocab(qids, docids, benchmark.topics[benchmark.query_type])


def test_bagofwords_create(monkeypatch, tmpdir, dummy_index):
    benchmark = DummyBenchmark({})
    extractor = BagOfWords(
        {"name": "bagofwords", "datamode": "unigram", "maxqlen": 4, "maxdoclen": 800, "usecache": False},
        provide={"index": dummy_index},
    )
    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics["title"])
    assert extractor.stoi == {
        "<pad>": 0,
        "dummy": 1,
        "doc": 2,
        "hello": 3,
        "world": 4,
        "greetings": 5,
        "from": 6,
        "outer": 7,
        "space": 8,
        "lessdummy": 9,
    }

    assert extractor.itos == {v: k for k, v in extractor.stoi.items()}
    assert extractor.embeddings == {
        "<pad>": 0,
        "dummy": 1,
        "doc": 2,
        "hello": 3,
        "world": 4,
        "greetings": 5,
        "from": 6,
        "outer": 7,
        "space": 8,
        "lessdummy": 9,
    }


def test_bagofwords_create_trigrams(monkeypatch, tmpdir, dummy_index):
    benchmark = DummyBenchmark({})
    tok_cfg = {"name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor = BagOfWords(
        {"name": "bagofwords", "datamode": "trigram", "maxqlen": 4, "maxdoclen": 800, "usecache": False},
        provide={"index": dummy_index, "tokenizer": tokenizer},
    )
    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics["title"])
    assert extractor.stoi == {
        "<pad>": 0,
        "#du": 1,
        "dum": 2,
        "umm": 3,
        "mmy": 4,
        "my#": 5,
        "#do": 6,
        "doc": 7,
        "oc#": 8,
        "#he": 9,
        "hel": 10,
        "ell": 11,
        "llo": 12,
        "lo#": 13,
        "#wo": 14,
        "wor": 15,
        "orl": 16,
        "rld": 17,
        "ld#": 18,
        "#gr": 19,
        "gre": 20,
        "ree": 21,
        "eet": 22,
        "eti": 23,
        "tin": 24,
        "ing": 25,
        "ngs": 26,
        "gs#": 27,
        "#fr": 28,
        "fro": 29,
        "rom": 30,
        "om#": 31,
        "#ou": 32,
        "out": 33,
        "ute": 34,
        "ter": 35,
        "er#": 36,
        "#sp": 37,
        "spa": 38,
        "pac": 39,
        "ace": 40,
        "ce#": 41,
        "#le": 42,
        "les": 43,
        "ess": 44,
        "ssd": 45,
        "sdu": 46,
    }

    assert extractor.itos == {v: k for k, v in extractor.stoi.items()}


def test_bagofwords_id2vec(tmpdir, dummy_index):
    benchmark = DummyBenchmark({})
    tok_cfg = {"name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor = BagOfWords(
        {"name": "bagofwords", "datamode": "unigram", "maxqlen": 4, "maxdoclen": 800, "usecache": False},
        provide={"index": dummy_index, "tokenizer": tokenizer},
    )
    extractor.stoi = {extractor.pad_tok: extractor.pad}
    extractor.itos = {extractor.pad: extractor.pad_tok}
    extractor.idf = defaultdict(lambda: 0)
    # extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics["title"])

    extractor.qid2toks = {"301": ["dummy", "doc"]}
    extractor.stoi["dummy"] = 1
    extractor.stoi["doc"] = 2
    extractor.itos[1] = "dummy"
    extractor.itos[2] = "doc"
    extractor.docid2toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0002": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    transformed = extractor.id2vec("301", "LA010189-0001", "LA010189-0001")
    # stoi only knows about the word 'dummy' and 'doc'. So the transformation of every other word is set as 0

    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] == "LA010189-0001"
    assert np.array_equal(transformed["query"], [0, 1, 1])
    assert np.array_equal(transformed["posdoc"], [6, 3, 0])
    assert np.array_equal(transformed["negdoc"], [6, 3, 0])
    assert np.array_equal(transformed["query_idf"], [0, 0, 0])


def test_bagofwords_id2vec_trigram(tmpdir, dummy_index):
    benchmark = DummyBenchmark({})
    tok_cfg = {"name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor = BagOfWords(
        {"name": "bagofwords", "datamode": "trigram", "maxqlen": 4, "maxdoclen": 800, "usecache": False},
        provide={"index": dummy_index, "tokenizer": tokenizer},
    )
    extractor.stoi = {extractor.pad_tok: extractor.pad}
    extractor.itos = {extractor.pad: extractor.pad_tok}
    extractor.idf = defaultdict(lambda: 0)
    # extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics["title"])

    extractor.qid2toks = {"301": ["dummy", "doc"]}
    extractor.docid2toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0002": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    extractor.stoi["#du"] = 1
    extractor.stoi["dum"] = 2
    extractor.stoi["umm"] = 3
    extractor.itos[1] = "#du"
    extractor.itos[2] = "dum"
    extractor.itos[3] = "umm"
    transformed = extractor.id2vec("301", "LA010189-0001")

    # stoi only knows about the word 'dummy'. So the transformation of every other word is set as 0
    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed.get("negdocid") is None

    # Right now we have only 3 words in the vocabular - "<pad>", "dummy" and "doc"
    assert np.array_equal(transformed["query"], [5, 1, 1, 1])
    assert np.array_equal(
        transformed["posdoc"], [39, 3, 3, 3]
    )  # There  are 6 unknown words in the doc, so all of them is encoded as 0
    assert np.array_equal(transformed["query_idf"], [0, 0, 0, 0])

    # Learn another word
    extractor.stoi["mmy"] = 4
    extractor.stoi["my#"] = 5
    extractor.stoi["#he"] = 6
    extractor.itos[4] = "mmy"
    extractor.itos[5] = "my#"
    extractor.itos[6] = "#he"

    transformed = extractor.id2vec("301", "LA010189-0001")
    # The posdoc transformation changes to reflect the new word
    assert np.array_equal(transformed["posdoc"], [32, 3, 3, 3, 3, 3, 1])


def test_bagofwords_caching(dummy_index, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(SlowEmbedText, "_load_pretrained_embeddings", fake_magnitude_embedding)

    extractor_cfg = {"name": "bagofwords", "datamode": "trigram", "maxqlen": 4, "maxdoclen": 800, "usecache": True}
    extractor = BagOfWords(extractor_cfg, provide={"index": dummy_index})

    benchmark = DummyBenchmark()

    qids = list(benchmark.qrels.keys())  # ["301"]
    qid = qids[0]
    docids = list(benchmark.qrels[qid].keys())

    assert not extractor.is_state_cached(qids, docids)

    extractor.preprocess(qids, docids, benchmark.topics[benchmark.query_type])

    assert extractor.is_state_cached(qids, docids)

    new_extractor = BagOfWords(extractor_cfg, provide={"index": dummy_index})

    assert new_extractor.is_state_cached(qids, docids)
    new_extractor._build_vocab(qids, docids, benchmark.topics[benchmark.query_type])


nltk.download("stopwords")


def test_deeptiles_extract_segment_long_text(tmpdir, monkeypatch, dummy_index):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    # nltk.TextTilingTokenizer only works with large blobs of text
    ttt = TextTilingTokenizer(k=6)
    extractor_config = {
        "name": "deeptiles",
        "embeddings": "glove6b",
        "tilechannels": 3,
        "passagelen": 30,
        "slicelen": 20,
        "tfchannel": True,
    }
    extractor = DeepTileExtractor(extractor_config, provide={"index": dummy_index})

    # blob of text with Shakespeare and Shangri La. Should split into two topics
    s = (
        "O that we now had here but one ten thousand of those men in England That do no work to-day. Whats he that "
        "wishes so? My cousin, Westmorland? No, my fair cousin. If we are marked to die, we are enough To do our "
        "country loss; and if to live, The fewer men, the greater share of honour. Gods will! I pray thee, wish"
        " not one man more. Shangri-La is a fictional place described in the 1933 novel Lost Horizon "
        "by British author James Hilton. Hilton describes Shangri-La as a mystical, harmonious valley, gently guided "
        "from a lamasery, enclosed in the western end of the Kunlun Mountains. Shangri-La has become synonymous with "
        "any earthly paradise, particularly a mythical Himalayan utopia – a permanently happy land, isolated from "
        "the world"
    )
    doc_toks = s.split(" ")
    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 2

    # The split was determined by nltk.TextTilingTokenizer. Far from perfect
    assert segments == [
        "O that we now had here but one ten thousand of those men in England That do no work to-day. Whats he that wishes so? My cousin, Westmorland? No, my fair cousin. If we are marked to die, we are",
        " enough To do our country loss; and if to live, The fewer men, the greater share of honour. Gods will! I pray thee, wish not one man more. Shangri-La is a fictional place described in the 1933 novel Lost Horizon by British author James Hilton. Hilton describes Shangri-La as a mystical, harmonious valley, gently guided from a lamasery, enclosed in the western end of the Kunlun Mountains. Shangri-La has become synonymous with any earthly paradise, particularly a mythical Himalayan utopia – a permanently happy land, isolated from the world",
    ]


def test_deeptiles_extract_segment_short_text(tmpdir, monkeypatch, dummy_index):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    # The text is too short for TextTilingTokenizer. Test if the fallback works
    ttt = TextTilingTokenizer(k=6)
    pipeline_config = {
        "name": "deeptiles",
        "passagelen": 30,
        "slicelen": 20,
        "tfchannel": True,
        "tilechannels": 3,
        "index": {"collection": {"name": "dummy"}},
    }
    extractor = DeepTileExtractor(pipeline_config, provide={"index": dummy_index})
    s = "But we in it shall be rememberèd We few, we happy few, we band of brothers"
    doc_toks = s.split(" ")
    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 1
    # N.B - segments are in all lowercase, special chars (comma) have been removed
    assert segments == ["But we in it shall be rememberèd We few, we happy few, we band of brothers"]

    s = (
        "But we in it shall be rememberèd We few, we happy few, we band of brothers. For he to-day that sheds his "
        "blood with me Shall be my brother"
    )
    doc_toks = s.split(" ")

    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 2
    assert segments == [
        "But we in it shall be rememberèd We few, we happy few, we band of brothers. For he to-day that",
        "sheds his blood with me Shall be my brother",
    ]


def test_deeptiles_clean_segments(tmpdir, dummy_index):
    pipeline_config = {"name": "deeptiles", "passagelen": 30, "slicelen": 20, "tfchannel": True, "tilechannels": 3}
    extractor = DeepTileExtractor(pipeline_config, provide={"index": dummy_index})
    assert extractor.clean_segments(["hello world", "foo bar"], p_len=4) == [
        "hello world",
        "foo bar",
        extractor.pad_tok,
        extractor.pad_tok,
    ]
    assert extractor.clean_segments(["hello world", "foo bar", "alice", "bob"], p_len=3) == ["hello world", "foo bar", "alicebob"]


def test_deeptiles_create_visualization_matrix(monkeypatch, tmpdir, dummy_index):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    pipeline_config = {"name": "deeptiles", "tilechannels": 3, "maxqlen": 5, "passagelen": 3, "slicelen": 20, "tfchannel": True}
    extractor = DeepTileExtractor(pipeline_config, provide={"index": dummy_index})
    extractor.stoi = {"<pad>": 0, "hello": 1, "world": 2, "foo": 3, "bar": 4, "alice": 5, "bob": 6}
    extractor.idf = defaultdict(lambda: 0)

    extractor._build_embedding_matrix()
    embeddings_matrix = extractor.embeddings
    level = extractor.create_visualization_matrix(
        ["hello", extractor.pad_tok, extractor.pad_tok, extractor.pad_tok, extractor.pad_tok],
        ["hello world", "foo bar", "alice bob"],
        embeddings_matrix,
    )
    assert level.shape == (1, 5, 3, 3)


def test_deeptiles_create(monkeypatch, tmpdir, dummy_index):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    benchmark = DummyBenchmark({})
    extractor_config = {
        "name": "deeptiles",
        "tilechannels": 3,
        "maxqlen": 5,
        "passagelen": 3,
        "slicelen": 20,
        "tfchannel": True,
        "embeddings": "glove6b",
        "usecache": False,
    }
    extractor = DeepTileExtractor(extractor_config, provide={"index": dummy_index})

    print("BT:", benchmark.topics["title"])
    extractor.preprocess(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics["title"])
    assert extractor.stoi == {
        "<pad>": 0,
        "dummy": 1,
        "doc": 2,
        "hello": 3,
        "world": 4,
        "greetings": 5,
        "from": 6,
        "outer": 7,
        "space": 8,
        "lessdummy": 9,
    }

    assert extractor.itos == {v: k for k, v in extractor.stoi.items()}
    assert extractor.stoi == {
        "<pad>": 0,
        "dummy": 1,
        "doc": 2,
        "hello": 3,
        "world": 4,
        "greetings": 5,
        "from": 6,
        "outer": 7,
        "space": 8,
        "lessdummy": 9,
    }
