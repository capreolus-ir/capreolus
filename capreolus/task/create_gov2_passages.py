from tqdm import tqdm
import random
import os
import math
import nltk
from capreolus import ConfigOption, Dependency
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class Gov2Passages(Task):
    module_name = "gov2passages"
    requires_random_seed = False
    config_spec = [
        ConfigOption("passagelen", 5, "Number of sentences that should be present in a passage"),
        ConfigOption("overlap", 2, "Number of sentences that should overlap between adjacent passages"),
        ConfigOption("output", None, "The path to the directory where the raw passages should be written to")
    ]

    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="gov2", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"})
    ]

    commands = ["generate"] + Task.help_commands
    default_command = "generate"

    def generate(self):
        # Take care to configure the index to keep the stopwords and not to step while invoking this task
        self.index.create_index()
        all_docids = self.get_all_docids_in_collection()
        passagelen = self.config["passagelen"]
        overlap = self.config["overlap"]
        output_dir = os.path.join(self.get_cache_path(), "generated") if self.config["output"] is None else self.config["output"]
        os.makedirs(output_dir, exist_ok=True)
        files_count = 0
        fout = open(os.path.join(output_dir, "collection_{}.txt".format(files_count)), "w", encoding="utf-8")
        total_psg_count = 0
        docs_count = 0

        for i, docid in tqdm(enumerate(all_docids), desc="generating"):
            docs_count += 1
            if i % 5000000 == 0:
                fout.close()
                files_count += 1
                fout = open(os.path.join(output_dir, "collection_{}.txt".format(files_count)), "w", encoding="utf-8")

            doc = self.index.get_doc(docid)
            sentences = nltk.sent_tokenize(doc)
            start_idx = 0
            passage_idx = 0

            passages = []
            while start_idx < len(sentences):
                selected_sentences = sentences[start_idx: start_idx + passagelen]
                passage = " ".join(selected_sentences)
                passage_id = "{}_{}".format(docid, passage_idx)
                trec_passage = document_to_trectxt(passage_id, passage)
                passages.append(trec_passage)
                fout.write(trec_passage)

                passage_idx += 1
                start_idx += passagelen - overlap

            total_psg_count += len(passages)
            if random.random() > 0.95:
                logger.debug("docid is {}".format(docid))
                logger.debug("There are {} sentences in the doc".format(len(sentences)))
                logger.debug("The original document is: {}".format(doc))
                logger.debug("The document generated {} passages".format(passage_idx + 1))
                formatted_passages = "\n----------\n".join(passages)
                logger.debug("The passages are: {}".format(formatted_passages))

        logger.info("Generated {} passages from {} gov2 docs".format(total_psg_count, docs_count))
        fout.close()

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
