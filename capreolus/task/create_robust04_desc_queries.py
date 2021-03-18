from collections import defaultdict
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
nltk.download('stopwords')

@Task.register
class Robust04DescQueries(Task):
    """
    We use DocT5 to generate queries for robust04passages and remove the stopwords
    """
    module_name = "robust04descqueries"
    requires_random_seed = True
    config_spec = [
        ConfigOption("querylen", 64, "DocT5 max query len parameter"),
        ConfigOption("keepstopwords", True, "Should stop words be removed"),
        ConfigOption("numqueries", 3, "How many queries need to be generated per passage?"),
        ConfigOption("maxqueriesperdoc",30, "The maximum number of queries generated per doc"),
        ConfigOption("queryoutput", "/home/kjose/capreolus/capreolus/data/topics.robust04doct5desc.txt"),
        ConfigOption("qrelsoutput", "/home/kjose/capreolus/capreolus/data/qrels.robust04doct5desc.txt"),
        ConfigOption("foldsoutput", "/home/kjose/capreolus/capreolus/data/robust04doct5desc.folds.json"),
        ConfigOption("doct5", "/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/docT5", "path to docT5 model"),

    ]

    dependencies = [
       Dependency(
           key="benchmark", module="benchmark", name="robust04passages", provide_children=["collection"]
       ),
       Dependency(
           key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
       ),
       Dependency(
           key="searcher", module="searcher", name="BM25"
       )

    ]

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate_queries(self, bm25_run):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        config = T5Config.from_pretrained('t5-base')
        t5_model = torch.nn.DataParallel(T5ForConditionalGeneration.from_pretrained(
            '/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/docT5/model.ckpt-1004000', from_tf=True, config=config))
        t5_model.to(device)

        qrels = self.benchmark.qrels
        relevant_docid_to_query = defaultdict(list)
        relevant_docids_in_qrels = set()
        for qid, docid_to_label in qrels.items():
            for docid, label in docid_to_label.items():
                if label >= 1:
                    relevant_docid_to_query[docid].append(qid)
                    relevant_docids_in_qrels.add(docid)

        docid_to_passageids = defaultdict(list)
        for qid, passage_id_to_score in bm25_run.items():
            for passage_id, score in passage_id_to_score.items():
                doc_id = passage_id.split("_")[0]
                docid_to_passageids[doc_id].append(passage_id)

        doc_to_generated_queries = defaultdict(lambda: 0)
        passage_to_generated_queries = defaultdict(list)
        for doc_id, qid_list in tqdm(relevant_docid_to_query.items(), desc="Generate queries"):
            for qid in qid_list:
                query_desc = self.benchmark.topics["desc"][qid]
                passages_in_doc = docid_to_passageids[doc_id]

                for passage_id in passages_in_doc:
                    if doc_to_generated_queries[passage_id.split("_")[0]] >= self.config["maxqueriesperdoc"]:
                        continue

                    passage = self.index.get_doc(passage_id)
                    input_ids = tokenizer.encode(passage + "</s>", return_tensors='pt').to(device)
                    output = t5_model.module.generate(input_ids=input_ids, max_length=self.config["querylen"], do_sample=True, top_k=10,
                                               num_return_sequences=self.config["numqueries"])
                    generated_queries = ["{} {}".format(query_desc, tokenizer.decode(output[i], skip_special_tokens=True)) for i in
                                         range(self.config["numqueries"])]

                    if not self.config["keepstopwords"]:
                        cleaned_queries = [" ".join([term for term in word_tokenize(query) if term not in stopwords.words()]) for
                                           query in generated_queries]
                    else:
                        cleaned_queries = generated_queries

                    passage_to_generated_queries[passage_id] = cleaned_queries
                    doc_to_generated_queries[passage_id.split("_")[0]] += len(cleaned_queries)

        return passage_to_generated_queries

    def generate_topics_and_qrels(self, passage_to_generated_queries):
        topic_id_offset = 1000
        generated_topics = {}
        generated_qrels = {}

        for passage_id, queries in passage_to_generated_queries.items():
            for query in queries:
                topic_id = str(topic_id_offset + len(generated_topics))
                generated_topics[topic_id] = query
                # The generated qrels will only have relevant docs.
                generated_qrels[topic_id] = {passage_id: 1}

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
        bm25_results_dir = self.search()
        bm25_run_fn = os.path.join(bm25_results_dir, "searcher")
        bm25_run = Searcher.load_trec_run(bm25_run_fn)
        passage_to_generated_queries = self.generate_queries(bm25_run)
        generated_topics, generated_qrels = self.generate_topics_and_qrels(passage_to_generated_queries)

        # Everything is a title query. topic_to_trectxt automatically uses title as desc if desc is not provided
        # Include the original topics as well in the topics file - else BM25 search won't use those queries
        topics = copy(self.benchmark.topics)
        for qid, query in generated_topics.items():
            topics["title"][qid] = query


        qrels = defaultdict(dict)
        for qid, docid_to_label in generated_qrels.items():
            qrels[qid] = copy(docid_to_label)

        folds = copy(self.benchmark.folds)
        for s in folds:
            folds[s]["train_qids"] = list(generated_topics.keys())

        self.write_to_file(topics, qrels, folds)

    def search(self):
        topics_fn = self.benchmark.topic_file
        output_dir = self.get_results_path()

        if hasattr(self.searcher, "index"):
            # All anserini indexes ignore the "fold" parameter. This is required for FAISS though, since we have to train an encoder
            self.searcher.index.create_index()

        search_results_folder = self.searcher.query_from_file(topics_fn, output_dir)
        logger.info("searcher results written to: %s", search_results_folder)

        return search_results_folder

