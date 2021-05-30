import json

import ir_datasets

from capreolus import ConfigOption
from capreolus.benchmark import Benchmark, IRDBenchmark
from capreolus.collection import Collection, IRDCollection


def dataset_to_collection(name):
    # adapted from https://github.com/Georgetown-IR-Lab/OpenNIR/blob/master/onir/datasets/irds.py#L47
    # HACK: find "parent" dataset that contains same docs handler so we don't re-build the index for the same collection
    ds = ir_datasets.load(name)
    segments = name.split("/")
    docs_handler = ds.docs_handler()
    parent_docs_ds = name
    while len(segments) > 1:
        segments = segments[:-1]
        parent_ds = ir_datasets.load("/".join(segments))
        if parent_ds.has_docs() and parent_ds.docs_handler() == docs_handler:
            parent_docs_ds = "/".join(segments)
    return parent_docs_ds


def get_irds(dataset, query_type, fields):
    if isinstance(fields, str):
        fields = [fields]

    if isinstance(dataset, str):
        dataset = [dataset]

    collection_datasets = {dataset_to_collection(name) for name in dataset}
    assert len(collection_datasets) == 1
    collection_dataset = list(collection_datasets)[0]

    @Collection.register
    class DynamicIRDCollection(IRDCollection):
        module_name = collection_dataset
        ird_dataset_name = collection_dataset
        config_spec = [ConfigOption("fields", ["body"], "fields to index", value_type="strlist")]
        collection_type = "JsonCollection"

        def doc_as_json(self, doc):
            content = " ".join((getattr(doc, field) for field in self.config["fields"]))
            return json.dumps({"id": doc.doc_id, "contents": content})

    @Benchmark.register
    class DynamicIRDBenchmark(IRDBenchmark):
        module_name = ",".join(dataset)
        config_spec = [ConfigOption("query_type", "title")]

        @property
        def query_type(self):
            return self.config["query_type"]

        @property
        def queries(self):
            return self.topics[self.query_type]

    return DynamicIRDCollection({"fields": fields}), DynamicIRDBenchmark({"query_type": query_type})
