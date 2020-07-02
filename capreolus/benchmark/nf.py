import json
import re

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

from . import Benchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class NF(Benchmark):
    """ NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval [1]

    [1] Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler. A Full-Text Learning to Rank Dataset for Medical Information Retrieval Proceedings of the 38th European Conference on Information Retrieval (ECIR), Padova, Italy, 2016. https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/
    """

    module_name = "nf"
    dependencies = [Dependency(key="collection", module="collection", name="nf")]
    config_spec = [
        ConfigOption(key="labelrange", default_value="0-2", description="range of dataset qrels, options: 0-2, 1-3"),
        ConfigOption(
            key="fields",
            default_value="all_titles",
            description="query fields included in topic file, "
            "options: 'all_fields', 'all_titles', 'nontopics', 'vid_title', 'vid_desc'",
        ),
    ]

    fold_file = PACKAGE_PATH / "data" / "nf.json"

    query_type = "title"

    def build(self):
        fields, label_range = self.config["fields"], self.config["labelrange"]
        self.field2kws = {
            "all_fields": ["all"],
            "nontopics": ["nontopic-titles"],
            "vid_title": ["vid-titles"],
            "vid_desc": ["vid-desc"],
            "all_titles": ["titles", "vid-titles", "nontopic-titles"],
        }
        self.labelrange2kw = {"0-2": "2-1-0", "1-3": "3-2-1"}

        if fields not in self.field2kws:
            raise ValueError(f"Unexpected fields value: {fields}, expect: {', '.join(self.field2kws.keys())}")
        if label_range not in self.labelrange2kw:
            raise ValueError(f"Unexpected label range: {label_range}, expect: {', '.join(self.field2kws.keys())}")

        self.qrel_file = PACKAGE_PATH / "data" / f"qrels.nf.{label_range}.txt"
        self.test_qrel_file = PACKAGE_PATH / "data" / f"test.qrels.nf.{label_range}.txt"
        self.topic_file = PACKAGE_PATH / "data" / f"topics.nf.{fields}.txt"
        self.download_if_missing()

    def _transform_qid(self, raw):
        """ NFCorpus dataset specific, remove prefix in query id since anserini convert all qid to integer """
        return raw.replace("PLAIN-", "")

    def download_if_missing(self):
        if all([f.exists() for f in [self.topic_file, self.fold_file, self.qrel_file]]):
            return

        tmp_corpus_dir = self.collection.download_raw()
        topic_f = open(self.topic_file, "w", encoding="utf-8")
        qrel_f = open(self.qrel_file, "w", encoding="utf-8")
        test_qrel_f = open(self.test_qrel_file, "w", encoding="utf-8")

        set_names = ["train", "dev", "test"]
        folds = {s: set() for s in set_names}
        qrel_kw = self.labelrange2kw[self.config["labelrange"]]
        for set_name in set_names:
            with open(tmp_corpus_dir / f"{set_name}.{qrel_kw}.qrel") as f:
                for line in f:
                    line = self._transform_qid(line)
                    qid = line.strip().split()[0]
                    folds[set_name].add(qid)
                    if set_name == "test":
                        test_qrel_f.write(line)
                    qrel_f.write(line)

            files = [tmp_corpus_dir / f"{set_name}.{keyword}.queries" for keyword in self.field2kws[self.config["fields"]]]
            qids2topics = self._align_queries(files, "title")

            for qid, txts in qids2topics.items():
                topic_f.write(topic_to_trectxt(qid, txts["title"]))

        json.dump(
            {"s1": {"train_qids": list(folds["train"]), "predict": {"dev": list(folds["dev"]), "test": list(folds["test"])}}},
            open(self.fold_file, "w"),
        )

        topic_f.close()
        qrel_f.close()
        test_qrel_f.close()
        logger.info(f"nf benchmark prepared")

    def _align_queries(self, files, field, qid2queries=None):
        if not qid2queries:
            qid2queries = {}
        for fn in files:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    qid, txt = line.strip().split("\t")
                    qid = self._transform_qid(qid)
                    txt = " ".join(re.sub("[^A-Za-z]", " ", txt).split()[:1020])
                    if qid not in qid2queries:
                        qid2queries[qid] = {field: txt}
                    else:
                        if field in qid2queries[qid]:
                            logger.warning(f"Overwriting title for query {qid}")
                        qid2queries[qid][field] = txt
        return qid2queries
