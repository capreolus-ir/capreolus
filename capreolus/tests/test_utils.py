from capreolus import constants
from capreolus.utils.trec import load_trec_topics

PACKAGE_PATH = constants["PACKAGE_PATH"]


def test_trec_topic_loader():
    dummy_topic = PACKAGE_PATH / "data" / "topics.dummy.style-2.txt"
    topics = load_trec_topics(dummy_topic)
    print(topics)

    assert topics["title"] == {"301": "Dummy doc", "302": "title of Dummy doc 302", "303": "title of Dummy doc 303"}
    assert topics["desc"] == {"301": "A dummy doc", "302": "desc of dummy doc 302", "303": "Description of dummy doc 303"}
    assert topics["narr"] == {
        "301": "The doc of the dummies",
        "302": "narr of The doc of the dummies 302",
        "303": "Narrative of the dummies doc 303",
    }
