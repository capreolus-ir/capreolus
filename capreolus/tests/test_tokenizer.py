from capreolus.index import AnseriniIndex
from capreolus.tokenizer import AnseriniTokenizer


def test_anserini_tokenzier():
    cfg = {"name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(cfg)

    index = AnseriniIndex({"name": "anserini", "indexstops": False, "stemmer": "porter", "collection": {"name": "dummy"}})
    index.create_index()

    docs = index.get_docs(["LA010189-0001"])
    print(tokenizer.tokenize(docs))
