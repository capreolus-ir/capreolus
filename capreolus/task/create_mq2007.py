from collections import defaultdict
import pickle
import json
from capreolus import Dependency, ConfigOption
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

import ir_datasets

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class MQ2007(Task):
    """
    Creates the MQ2007 collection topics, qrels and folds. The test fold of MQ2007 here is designed to be the same as
    that of GOV2 - only the train fold differs.
    """

    module_name = "mq2007"
    requires_random_seed = True
    config_spec = [
        ConfigOption("queryoutput", "/home/kjose/capreolus/capreolus/data/topics.mq2007.txt"),
        ConfigOption("qrelsoutput", "/home/kjose/capreolus/capreolus/data/qrels.mq2007.txt"),
        ConfigOption("foldsoutput", "/home/kjose/capreolus/capreolus/data/mq2007.folds.json"),
    ]

    dependencies = []

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate(self):
        mq2007_dataset = ir_datasets.load("gov2/trec-mq-2007")
        tb04_dataset = ir_datasets.load("gov2/trec-tb-2004")
        tb05_dataset = ir_datasets.load("gov2/trec-tb-2005")
        tb06_dataset = ir_datasets.load("gov2/trec-tb-2006")

        old_queries = set()
        gov2_queries = []
        gov2_query_to_qid = {}

        for query in tb04_dataset.queries_iter():
            old_queries.add(query.title.lower())
            gov2_query_to_qid[query.title.lower()] = query.query_id
            gov2_queries.append(query)

        for query in tb05_dataset.queries_iter():
            old_queries.add(query.title.lower())
            gov2_query_to_qid[query.title.lower()] = query.query_id
            gov2_queries.append(query)

        for query in tb06_dataset.queries_iter():
            old_queries.add(query.title.lower())
            gov2_query_to_qid[query.title.lower()] = query.query_id
            gov2_queries.append(query)

        assert len(old_queries) == 150

        old_qrels = defaultdict(dict)
        self.update_qrels(old_qrels, tb04_dataset)
        self.update_qrels(old_qrels, tb05_dataset)
        self.update_qrels(old_qrels, tb06_dataset)

        duplicate_queries = []
        new_queries = []
        mq2007_qid_to_query = {}
        for query in mq2007_dataset.queries_iter():
            if query.text.lower() in old_queries:
                duplicate_queries.append(query)
            else:
                new_queries.append(query)

            mq2007_qid_to_query[query.query_id] = query.text.lower()

        gov2_qid_to_mq_qid = {}
        for mq_qid, query_text in mq2007_qid_to_query.items():
            gov2_qid = gov2_query_to_qid.get(query_text)
            if gov2_qid:
                gov2_qid_to_mq_qid[gov2_qid] = mq_qid

        assert len(duplicate_queries) == 150
        duplicate_qrels = 0
        deeper_qrels = 0
        new_qrels = 0

        duplicate_qrel_qids = set()
        deeper_qrel_qids = set()
        new_qrel_qids = set()

        for qrel in mq2007_dataset.qrels_iter():
            qid = qrel.query_id
            doc_id = qrel.doc_id
            query_text = mq2007_qid_to_query[qid]
            gov2_qid = gov2_query_to_qid.get(query_text)

            if query_text in old_queries and doc_id in old_qrels[gov2_qid]:
                duplicate_qrels += 1
                duplicate_qrel_qids.add(qrel.query_id)
            elif query_text in old_qrels and doc_id not in old_qrels[gov2_qid]:
                deeper_qrels += 1
                deeper_qrel_qids.add(qrel.query_id)
            elif qid not in old_qrels:
                new_qrels += 1
                new_qrel_qids.add(qrel.query_id)

        mq_qids_with_qrels = duplicate_qrel_qids.union(deeper_qrel_qids).union(new_qrel_qids)
        logger.info("There are {} MQ qids with qrels".format(len(mq_qids_with_qrels)))
        logger.info(
            "There are {} duplicate judgements, {} deeper judgements, and {} new judgements".format(
                duplicate_qrels, deeper_qrels, new_qrels
            )
        )
        logger.info(
            "There are {} qids with duplicate judgements, {} qids with deeper judgements, and {} with new judgements".format(
                len(duplicate_qrel_qids), len(deeper_qrel_qids), len(new_qrel_qids)
            )
        )
        logger.info("There are {} query duplicates and {} new queries".format(len(duplicate_queries), len(new_queries)))

        new_query_ids = [query.query_id for query in new_queries if query.query_id in mq_qids_with_qrels]
        gov2_query_ids = [query.query_id for query in list(duplicate_queries)]

        folds = {
            "s1": {
                "train_qids": new_query_ids,
                "predict": {
                    "dev": gov2_query_ids,
                    "test": gov2_query_ids,
                },
            }
        }
        with open(self.config["foldsoutput"], "w") as out_f:
            json.dump(folds, out_f)

        with open(self.config["queryoutput"], "w") as out_f:
            for query in new_queries + duplicate_queries:
                if query.query_id in mq_qids_with_qrels:
                    out_f.write(topic_to_trectxt(query.query_id, query.text.lower()))

        duplicate_query_ids_set = set([query.query_id for query in duplicate_queries])
        query_ids_that_should_be_copied = []
        with open(self.config["qrelsoutput"], "w") as out_f:
            for qrel in mq2007_dataset.qrels_iter():
                mq_query_id = qrel.query_id
                if mq_query_id in duplicate_query_ids_set:
                    query_ids_that_should_be_copied.append(mq_query_id)
                else:
                    out_f.write("{} 0 {} {}\n".format(qrel.query_id, qrel.doc_id, qrel.relevance))

            # For those queries that exist in gov2, copy over the qrels from gov2 instead of using mq2007's qrels
            for gov2_qid, docid_to_relevance in old_qrels.items():
                mq_qid = gov2_qid_to_mq_qid[gov2_qid]
                for docid, relevance in docid_to_relevance.items():
                    out_f.write("{} 0 {} {}\n".format(mq_qid, docid, relevance))

            # Qrels for gov2 - download it from here: https://lintool.github.io/Ivory/data/gov2/qrels.gov2.all

        pickle.dump(gov2_qid_to_mq_qid, open("/home/kjose/gov2_qid_to_mq_qid.dump", "wb"), protocol=-1)
        pickle.dump(old_qrels, open("/home/kjose/old_qrels.dump", "wb"), protocol=-1)
        pickle.dump(gov2_queries, open("/home/kjose/gov2_queries.dump", "wb"), protocol=-1)

    def update_qrels(self, qrels_dict, dataset):
        for qrel in dataset.qrels_iter():
            qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
