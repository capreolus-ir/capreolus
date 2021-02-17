import ir_datasets

from capreolus import Dependency, constants

from . import Benchmark, IRDBenchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class CDS(IRDBenchmark):
    module_name = "cds"
    query_type = "text"
    ird_dataset_names = ["pmc/v1/trec-cds-2014", "pmc/v1/trec-cds-2015", "pmc/v2/trec-cds-2016"]
    dependencies = [Dependency(key="collection", module="collection", name="cds")]
    fold_file = PACKAGE_PATH / "data" / "cds_5folds.json"
    query_type = "summary"
    query_types = {}  # diagnosis, treatment, or test

    def ird_load_qrels(self):
        qrels = {}
        for name in self.ird_dataset_names:
            year = name.split("-")[-1]
            assert len(year) == 4

            dataset = ir_datasets.load(name)
            for qrel in dataset.qrels_iter():
                qid = year + qrel.query_id
                qrels.setdefault(qid, {})
                qrels[qid][qrel.doc_id] = max(qrel.relevance, qrels[qid].get(qrel.doc_id, -1))

        return qrels

    def ird_load_topics(self):
        topics = {}
        field = "description" if self.query_type == "desc" else self.query_type

        for name in self.ird_dataset_names:
            year = name.split("-")[-1]
            assert len(year) == 4

            dataset = ir_datasets.load(name)
            for query in dataset.queries_iter():
                qid = year + query.query_id
                topics[qid] = getattr(query, field).replace("\n", " ")
                self.query_types[qid] = query.type

        return {self.query_type: topics}
