from collections import defaultdict
from pathlib import Path
import os
import json
from copy import copy

import torch
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from capreolus import ConfigOption, Dependency
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt


logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class Robust04SimmatQueries(Task):
    """
    We use DocT5 to generate queries for robust04passages and remove the stopwords
    """
    module_name = "robust04simmatqueries"
    requires_random_seed = True
    config_spec = [
        ConfigOption("maxqueriesperdoc", 10, "The maximum number of queries generated per doc"),
        ConfigOption("queryoutput", "/GW/NeuralIR/nobackup/kevin_cache/topics.robust04simmat.txt"),
        ConfigOption("qrelsoutput", "/GW/NeuralIR/nobackup/kevin_cache/qrels.robust04simmat.txt"),
        ConfigOption("foldsoutput", "/GW/NeuralIR/nobackup/kevin_cache/robust04simmat.folds.json"),
        ConfigOption("modeldir", "/GW/NeuralIR/nobackup/kevin_cache/cedr-knrm", "Path to a trained ELECTRA-KNRM model"),
        ConfigOption("threshold", 0.8, "The similarity score above which we consider the terms to be equal")
    ]

    dependencies = [
       Dependency(
           key="benchmark", module="benchmark", name="robust04.yang19", provide_children=["collection"]
       ),
       Dependency(
           key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
       ),
       Dependency(
           key="searcher", module="searcher", name="BM25"
       ),
       Dependency(
           key="reranker", module="reranker", name="CEDRKNRM", default_config_overrides={"cls": None, "simmat_layers": 12}
       )
    ]

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate_queries(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reranker.build_model()
        self.reranker.trainer.load_best_model(self.reranker, Path(self.config["modeldir"]))
        self.reranker.model.to(device)

        qrels = self.benchmark.qrels
        relevant_docid_to_query = defaultdict(list)
        relevant_docids_in_qrels = set()
        for qid, docid_to_label in qrels.items():
            for docid, label in docid_to_label.items():
                if label >= 1:
                    relevant_docid_to_query[docid].append(qid)
                    relevant_docids_in_qrels.add(docid)

        doc_to_generated_queries = defaultdict(lambda: 0)
        passage_to_generated_queries = defaultdict(list)
        self.reranker.extractor.preprocess(list(qrels.keys()), list(relevant_docids_in_qrels), self.benchmark.topics["title"])

        for doc_id, qid_list in tqdm(relevant_docid_to_query.items(), desc="Generate queries"):
            for qid in qid_list:
                input_data = self.reranker.extractor.id2vec(qid, doc_id, label=[1, 0])
                input_data = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in input_data.items()}
                candidates = self.reranker.get_replacement_candidates(input_data)
                logger.info("candidates are: {}".format(candidates))

                raise Exception("Bam!")
                # generate queries using the simmat
                # passage_to_generated_queries[passage_id] = cleaned_queries
                # doc_to_generated_queries[passage_id.split("_")[0]] += len(cleaned_queries)

        return passage_to_generated_queries

    def generate_topics_and_qrels(self, passage_to_generated_queries):
        topic_id_offset = 1000
        generated_topics = {}
        generated_qrels = {}

        for passage_id, queries in passage_to_generated_queries.items():
            doc_id = passage_id.split("_")[0]
            for query in queries:
                topic_id = str(topic_id_offset + len(generated_topics))
                generated_topics[topic_id] = query
                # The generated qrels will only have relevant docs.
                generated_qrels[topic_id] = {doc_id: 1}

        return generated_topics, generated_qrels

    def write_to_file(self, extended_topics, extended_qrels, extended_folds):
        with open(self.config["queryoutput"], "w") as out_f:
            for qid, query in extended_topics["title"].items():
                out_f.write(topic_to_trectxt(qid, query))

        with open(self.config["qrelsoutput"], "w") as out_f:
            for qid, docid_to_label in extended_qrels.items():
                for docid, label in docid_to_label.items():
                    out_f.write("{qid} 0 {docid} {label}\n".format(qid=qid, docid=docid, label=label))

        with open(self.config["foldsoutput"], "w") as out_f:
            json.dump(extended_folds, out_f)

    def generate(self):
        self.index.create_index()
        passage_to_generated_queries = self.generate_queries()
        generated_topics, generated_qrels = self.generate_topics_and_qrels(passage_to_generated_queries)

        logger.info(passage_to_generated_queries)

        extended_topics = copy(self.benchmark.topics)

        for qid, query in generated_topics.items():
            extended_topics["title"][qid] = query

        extended_qrels = copy(self.benchmark.qrels)
        for qid, docid_to_label in generated_qrels.items():
            extended_qrels[qid] = copy(docid_to_label)

        extended_folds = copy(self.benchmark.folds)
        for s in extended_folds:
            extended_folds[s]["train_qids"].extend(generated_topics.keys())

        self.write_to_file(extended_topics, extended_qrels, extended_folds)

    def search(self):
        topics_fn = self.benchmark.topic_file
        output_dir = self.get_results_path()

        if hasattr(self.searcher, "index"):
            # All anserini indexes ignore the "fold" parameter. This is required for FAISS though, since we have to train an encoder
            self.searcher.index.create_index()

        search_results_folder = self.searcher.query_from_file(topics_fn, output_dir)
        logger.info("searcher results written to: %s", search_results_folder)

        return search_results_folder

