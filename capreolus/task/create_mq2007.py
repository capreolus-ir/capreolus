from collections import defaultdict
import json
from capreolus import Dependency, ConfigOption
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

import ir_datasets

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class Robust04DescQueries(Task):
    """
    We use DocT5 to generate queries for robust04passages and remove the stopwords
    """
    module_name = "mq2007"
    requires_random_seed = True
    config_spec = [
        ConfigOption("queryoutput", "/home/kjose/capreolus/capreolus/data/topics.mq2007.txt"),
        ConfigOption("qrelsoutput", "/home/kjose/capreolus/capreolus/data/qrels.mq2007.txt"),
        ConfigOption("foldsoutput", "/home/kjose/capreolus/capreolus/data/mq2007.folds.json"),
        ConfigOption("gov2queryoutput", "/home/kjose/capreolus/capreolus/data/topics.gov2.txt"),
        ConfigOption("gov2foldsoutput", "/home/kjose/capreolus/capreolus/data/gov2.folds.json"),
    ]

    dependencies = [
    ]

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate(self):
        mq2007_dataset = ir_datasets.load("gov2/trec-mq-2007")
        tb04_dataset = ir_datasets.load("gov2/trec-tb-2004")
        tb05_dataset = ir_datasets.load("gov2/trec-tb-2005")
        tb06_dataset = ir_datasets.load("gov2/trec-tb-2006")

        old_queries = set()

        for query in tb04_dataset.queries_iter():
            old_queries.add(query.title.lower())

        for query in tb05_dataset.queries_iter():
            old_queries.add(query.title.lower())

        for query in tb06_dataset.queries_iter():
            old_queries.add(query.title.lower())

        assert len(old_queries) == 150

        old_qrels = defaultdict(dict)
        self.update_qrels(old_qrels, tb04_dataset)
        self.update_qrels(old_qrels, tb05_dataset)
        self.update_qrels(old_qrels, tb06_dataset)

        duplicate_qrels = 0
        deeper_qrels = 0
        new_qrels = 0

        for qrel in mq2007_dataset.qrels_iter():
            qid = qrel.query_id
            doc_id = qrel.doc_id

            if qid in old_qrels and doc_id in old_qrels[qid]:
                duplicate_qrels += 1
            elif qid in old_qrels and doc_id not in old_qrels[qid]:
                deeper_qrels += 1
            elif qid not in old_qrels:
                new_qrels += 1

        duplicate_queries = []
        new_queries = []
        for query in mq2007_dataset.queries_iter():
            if query.text in old_queries:
                duplicate_queries.append(query)
            else:
                new_queries.append(query)

        assert len(duplicate_queries) == 150
        logger.info("There are {} duplicate judgements, {} deeper judgements, and {} new judgements".format(duplicate_qrels, deeper_qrels, new_qrels))
        logger.info("There are {} query duplicates and {} new queries".format(len(duplicate_queries), len(new_queries)))

        new_query_ids = [query.query_id for query in new_queries]
        old_set_1 = [query.query_id for query in duplicate_queries[:50]]
        old_set_2 = [query.query_id for query in duplicate_queries[50:100]]
        old_set_3 = [query.query_id for query in duplicate_queries[100: 150]]

        folds = {
            "s1": {
                "train_qids": new_query_ids + old_set_1,
                "predict": {
                    "dev": old_set_2,
                    "test": old_set_3
                }
            },
            "s2": {
                "train_qids": new_query_ids + old_set_2,
                "predict": {
                    "dev": old_set_3,
                    "test": old_set_1
                }
            },
            "s3": {
                "train_qids": new_query_ids + old_set_3,
                "predict": {
                    "dev": old_set_1,
                    "test": old_set_2
                }
            }
        }
        with open(self.config["foldsoutput"], "w") as out_f:
            json.dump(folds, out_f)

        with open(self.config["queryoutput"], "w") as out_f:
            for query in new_queries:
                out_f.write(topic_to_trectxt(query.query_id, query.text.lower()))

        with open(self.config["qrelsoutput"], "w") as out_f:
            for qrel in mq2007_dataset.qrels_iter():
                out_f.write("{} 0 {} {}\n".format(qrel.query_id, qrel.doc_id, qrel.relevance))

        gov2_folds = {
            "s1": {
                "train_qids": old_set_1,
                "predict": {
                    "dev": old_set_2,
                    "test": old_set_3
                }
            },
            "s2": {
                "train_qids": old_set_2,
                "predict": {
                    "dev": old_set_3,
                    "test": old_set_1
                }
            },
            "s3": {
                "train_qids": old_set_3,
                "predict": {
                    "dev": old_set_1,
                    "test": old_set_2
                }
            }
        }

        with open(self.config["gov2foldsoutput"], "w") as out_f:
            json.dump(gov2_folds, out_f)

        with open(self.config["gov2queryoutput"], "w") as out_f:
            for query in duplicate_queries:
                out_f.write(topic_to_trectxt(query.query_id, query.text.lower()))

        # Qrels for gov2 - download it from here: https://lintool.github.io/Ivory/data/gov2/qrels.gov2.all

    def update_qrels(self, qrels_dict, dataset):
        for qrel in dataset.qrels_iter():
            qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
