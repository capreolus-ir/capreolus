from pytest_mock import mocker

from capreolus.collection import COLLECTIONS, Collection


def test_get_qid_from_query_string(mocker):
    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "Hello world"}}
        return collection._topics

    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]
    collection.topics
    q_id = collection.get_qid_from_query_string("foo")
    assert q_id is None

    q_id = collection.get_qid_from_query_string("Hello world")
    assert q_id == "q_1"


def test_get_relevance_query_string_not_part_of_collection(mocker):
    @property
    def mock_qrels(collection, *args):
        collection._qrels = {"q_1": {"doc_1": 1, "doc_2": 3, "doc_3": 4}, "q_2": {"doc_1": 2}}
        return collection._qrels

    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "Hello world"}}
        return collection._topics

    mocker.patch.object(Collection, "qrels", mock_qrels)
    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]
    collection.topics
    collection.qrels
    relevance = collection.get_relevance("foo", ["doc_1"])

    assert relevance == [0]


def test_get_relevance_doc_id_not_present_in_qrels(mocker):
    @property
    def mock_qrels(collection, *args):
        collection._qrels = {"q_1": {"doc_1": 1, "doc_2": 3, "doc_3": 4}, "q_2": {"doc_1": 2}}
        return collection._qrels

    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "Hello world", "q_2": "foo"}}
        return collection._topics

    mocker.patch.object(Collection, "qrels", mock_qrels)
    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]
    relevance = collection.get_relevance("Hello world", ["doc_34"])

    assert relevance == [0]


def test_get_relevance_doc_id_is_relevant(mocker):
    @property
    def mock_qrels(collection, *args):
        collection._qrels = {"q_1": {"doc_1": 1, "doc_2": 3, "doc_3": 4}, "q_2": {"doc_1": 2}}
        return collection._qrels

    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "Hello world", "q_2": "foo"}}
        return collection._topics

    mocker.patch.object(Collection, "qrels", mock_qrels)
    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]

    relevance = collection.get_relevance("Hello world", ["doc_1", "doc_3"])
    assert relevance == [1, 4]

    relevance = collection.get_relevance("foo", ["doc_1"])
    assert relevance == [2]


def test_get_relevance_doc_id_is_relevant_case_insensitive(mocker):
    @property
    def mock_qrels(collection, *args):
        collection._qrels = {"q_1": {"doc_1": 1, "doc_2": 3, "doc_3": 4}, "q_2": {"doc_1": 2}}
        return collection._qrels

    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "HeLLo world", "q_2": "fOO"}}
        return collection._topics

    mocker.patch.object(Collection, "qrels", mock_qrels)
    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]

    relevance = collection.get_relevance("Hello world", ["doc_1", "doc_3"])
    assert relevance == [1, 4]

    relevance = collection.get_relevance("foo", ["doc_1"])
    assert relevance == [2]


def test_get_collection_from_index_path():
    index_path_1 = "robust04/something/anserini/foo"
    index_path_2 = "/foo/bar"

    collection = Collection.get_collection_from_index_path(index_path_1)
    assert collection.name == "robust04"

    collection = Collection.get_collection_from_index_path(index_path_2)
    assert collection is None


def test_get_suggestions(mocker):
    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "HeLLo world", "q_2": "world", "q_3": "foo"}}
        return collection._topics

    mocker.patch.object(Collection, "topics", mock_topics)
    collection = COLLECTIONS["robust04"]

    suggestions = collection.get_query_suggestions("world", 1)
    assert suggestions == ["HeLLo world"]

    suggestions = collection.get_query_suggestions("world", 10)
    assert suggestions == ["HeLLo world", "world"]

    suggestions = collection.get_query_suggestions("foo", 10)
    assert suggestions == ["foo"]
