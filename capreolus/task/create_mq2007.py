from collections import defaultdict
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
            old_queries.add(query.title)

        for query in tb05_dataset.queries_iter():
            old_queries.add(query.title)

        for query in tb06_dataset.queries_iter():
            old_queries.add(query.title)

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

        logger.info("There are {} duplicates, {} deeper judgements, and {} new judgements".format(duplicate_qrels, deeper_qrels, new_qrels))

    def update_qrels(self, qrels_dict, dataset):
        for qrel in dataset.qrels_iter():
            qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
