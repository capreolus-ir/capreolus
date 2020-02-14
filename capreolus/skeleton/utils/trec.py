import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_ntcir_topics(fn):
    topics = {}

    tree = ET.parse(fn)
    for child in tree.getroot():
        qid = child.find("qid").text.strip()
        query = child.find("content").text.strip()

        assert qid not in topics
        assert len(qid) > 0 and len(query) > 0
        topics[qid] = query

    return {"content": topics}


def load_trec_topics(queryfn):
    title, desc, narr = defaultdict(list), defaultdict(list), defaultdict(list)

    block = None
    if queryfn.endswith(".gz"):
        openf = gzip.open
    else:
        openf = open

    with openf(queryfn, "rt") as f:
        for line in f:
            line = line.strip()

            if line.startswith("<num>"):
                # <num> Number: 700
                qid = line.split()[-1]
                # no longer an int
                # assert qid > 0
                block = None
            elif line.startswith("<title>"):
                # <title>  query here
                title[qid].extend(line.strip().split()[1:])
                block = "title"
                # TODO does this sometimes start with Topic: ?
                assert "Topic:" not in line
            elif line.startswith("<desc>"):
                # <desc> description \n description
                desc[qid].extend(line.strip().split()[1:])
                block = "desc"
            elif line.startswith("<narr>"):
                # same format as <desc>
                narr[qid].extend(line.strip().split()[1:])
                block = "narr"
            elif line.startswith("</top>") or line.startswith("<top>"):
                block = None
            elif block == "title":
                title[qid].extend(line.strip().split())
            elif block == "desc":
                desc[qid].extend(line.strip().split())
            elif block == "narr":
                narr[qid].extend(line.strip().split())

    out = {}
    if len(title) > 0:
        out["title"] = {qid: " ".join(terms) for qid, terms in title.items()}
    if len(desc) > 0:
        out["desc"] = {qid: " ".join(terms) for qid, terms in desc.items()}
    if len(narr) > 0:
        out["narr"] = {qid: " ".join(terms) for qid, terms in narr.items()}

    return out


def load_qrels(qrelfile, qids=None, include_spam=True):
    labels = defaultdict(dict)
    with open(qrelfile, "rt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            cols = line.split()
            qid, docid, label = cols[0], cols[2], int(cols[3])

            if qids is not None and qid not in qids:
                continue
            if label < 0 and not include_spam:
                continue

            labels[qid][docid] = label

    # remove qids with no relevant docs
    for qid in list(labels.keys()):
        if max(labels[qid].values()) <= 0:
            del labels[qid]

    labels.default_factory = None  # behave like normal dict
    return labels
