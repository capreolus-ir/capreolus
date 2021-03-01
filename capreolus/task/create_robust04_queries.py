from collections import defaultdict
import json
from copy import copy

import torch
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from capreolus import ConfigOption, Dependency
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt


logger = get_logger(__name__)  # pylint: disable=invalid-name
nltk.download('stopwords')

@Task.register
class Robust04Queries(Task):
    """
    We use DocT5 to generate queries for robust04passages and remove the stopwords
    """
    module_name = "robust04queries"
    requires_random_seed = True
    config_spec = [
        ConfigOption("querylen", 64, "DocT5 max query len parameter"),
        ConfigOption("numqueries", 3, "How many queries need to be generated per doc?"),
        ConfigOption("queryoutput", "/GW/NeuralIR/nobackup/kevin_cache/robust04doct5.topics.txt"),
        ConfigOption("qrelsoutput", "/GW/NeuralIR/nobackup/kevin_cache/robust04doct5.qrels.txt"),
        ConfigOption("foldsoutput", "/GW/NeuralIR/nobackup/kevin_cache/robust04doct5.folds.json"),
        ConfigOption("doct5", "/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/docT5", "path to docT5 model")
    ]

    dependencies = [
       Dependency(
           key="benchmark", module="benchmark", name="robust04passages", provide_children=["collection"]
       ),
       Dependency(
           key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
       )
    ]

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate_queries(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        config = T5Config.from_pretrained('t5-base')
        t5_model = T5ForConditionalGeneration.from_pretrained(
            '/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/docT5/model.ckpt-1004000', from_tf=True, config=config)
        t5_model.to(device)

        qrels = self.benchmark.qrels
        relevant_docids_in_qrels = set([docid for qid, docid_to_label in qrels.items() for docid, label in docid_to_label.items() if label >= 1])
        all_passageids = self.get_all_docids_in_collection()

        passage_ids_for_query_generation = [passage_id for passage_id in all_passageids if passage_id.split("_")[0] in relevant_docids_in_qrels]
        self.rng.shuffle(passage_ids_for_query_generation)

        passage_ids_for_query_generation = passage_ids_for_query_generation[:10]

        passage_to_generated_queries = defaultdict(list)
        doc_to_generated_queries = defaultdict(lambda: 0)

        for passage_id in tqdm(passage_ids_for_query_generation, desc="Generating"):
            # Generate only a fixed number of queries per doc
            if doc_to_generated_queries[passage_id.split("_")[0]] >= self.config["numqueries"] * 3:
                continue

            passage = self.index.get_doc(passage_id)
            input_ids = tokenizer.encode(passage + "</s>", return_tensors='pt').to(device)
            output = t5_model.generate(input_ids=input_ids, max_length=self.config["querylen"], do_sample=True, top_k=10, num_return_sequences=self.config["numqueries"])
            generated_queries = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(self.config["numqueries"])]
            cleaned_queries = [" ".join([term for term in word_tokenize(query) if term not in stopwords.words()]) for query in generated_queries]

            passage_to_generated_queries[passage_id] = cleaned_queries
            doc_to_generated_queries[passage_id.split("_")[0]] += len(cleaned_queries)

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
            for qid, query in extended_topics.items():
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

    def get_all_docids_in_collection(self):
        from jnius import autoclass
        anserini_index = self.index
        anserini_index.create_index()
        anserini_index_path = anserini_index.get_index_path().as_posix()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        # TODO: Add check for deleted rows in the index
        all_doc_ids = [anserini_index.convert_lucene_id_to_doc_id(i) for i in range(0, anserini_index_reader.maxDoc())]

        return all_doc_ids
