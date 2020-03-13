from capreolus.collection import DummyCollection
from capreolus.index import AnseriniIndex
from capreolus.tokenizer import AnseriniTokenizer


def test_anserini_tokenzier():
    cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(cfg)

    index = AnseriniIndex(
        {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    )
    index.modules["collection"] = DummyCollection({"_name": "dummy"})
    index.create_index()

    docs = index.get_docs(["LA010189-0001"])
    print(tokenizer.tokenize(docs))
