import gzip
import os
import shutil
import tarfile
import yaml

from capreolus.utils.common import download_file
from capreolus.utils.trec import load_qrels, load_trec_topics
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

_data_keys = ["topics", "qrels", "documents"]
_datadir = os.path.dirname(__file__)


def parse_collections():
    logger.debug("checking for collections to import in: %s", _datadir)
    d = {}
    for fn in os.listdir(_datadir):
        if fn.endswith(".yaml"):
            with open(os.path.join(_datadir, fn), "rt") as f:
                logger.debug("loading %s", fn)
                collection = Collection(yaml.load(f.read(), Loader=yaml.SafeLoader))
                d[collection.name] = collection
    return d


class Collection:
    IRRELEVANT = 0

    def __init__(self, config):
        Collection.validate_config(config)
        self.basepath = _datadir
        Collection.normalize_paths(config, self.basepath)
        self.name = config["name"]
        self.config = config
        self.is_large_collection = config.get("is_large_collection")

    def download_if_missing(self, cachedir):
        if os.path.exists(self.config["documents"]["path"]):
            return
        elif "index_download" not in self.config["documents"]:
            raise IOError(
                f"a download URL is not available for collection={self.name} and the collection path {self.config['documents']['path']} does not exist; you must manully place the document collection at this path in order to use this collection"
            )

        # Download the collection from URL and extract into a path in the cache directory.
        # To avoid re-downloading every call, we create an empty '/done' file in this directory on success.
        downloaded_collection_dir = os.path.join(cachedir, self.name, "downloaded")
        done_file = os.path.join(downloaded_collection_dir, "done")
        document_dir = os.path.join(downloaded_collection_dir, "documents")

        self.config["documents"]["path"] = document_dir
        # already downloaded?
        if os.path.exists(done_file):
            return True

        # 1. Download and extract Anserini index to a temporary location
        tmp_dir = os.path.join(downloaded_collection_dir, "tmp")
        archive_file = os.path.join(tmp_dir, "archive_file")
        os.makedirs(document_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info("downloading missing collection %s to temporary file %s", self.name, archive_file)
        download_file(
            self.config["documents"]["index_download"]["url"],
            archive_file,
            expected_hash=self.config["documents"]["index_download"]["sha256"],
        )

        logger.debug("extracting to %s", tmp_dir)
        with tarfile.open(archive_file) as tar:
            tar.extractall(path=tmp_dir)

        extracted_dir = os.path.join(tmp_dir, self.config["documents"]["index_download"]["index_directory_inside"])
        if not (os.path.exists(extracted_dir) and os.path.isdir(extracted_dir)):
            raise ValueError(f"could not find expected index directory {extracted_dir} in {tmp_dir}")

        # 2. Move Anserini index to its correct location in the cache
        index_config = self.config["documents"]["index_download"]["index_config_string"]
        index_dir = os.path.join(cachedir, self.name, index_config, "index")
        shutil.move(extracted_dir, index_dir)

        # 3. Extract raw documents from the Anserini index to document_dir
        index_to_trec_docs(index_dir, document_dir, self.config["documents"]["index_download"]["expected_document_count"])

        # remove temporary file and create a /done we can use to verify extraction was successful
        os.remove(archive_file)
        with open(done_file, "wt") as outf:
            print("", file=outf)

        logger.info("missing collection %s saved to %s", self.config["name"], document_dir)

    def set_qrels(self, path):
        self._qrels = load_qrels(path)

    def set_topics(self, path):
        self._topics = load_trec_topics(path)

    def set_documents(self, path):
        self.config["documents"]["path"] = path
 
    @property
    def qrels(self):
        if not hasattr(self, "_qrels"):
            self._qrels = load_qrels(self.config["qrels"]["path"])
        return self._qrels

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.config["topics"]["path"])
        return self._topics

    def get_qid_from_query_string(self, query_string):
        # TODO: This is a linear search across all topics (250 topics in Rob04). Verify that this doesn't add
        # significantly to the API response time
        for q_id, query in self.topics["title"].items():
            if query_string.lower() == query.lower():
                return q_id

        return None

    def get_relevance(self, query_string, doc_ids):
        """
        Get the relevance of a set of documents for a given query_string
        If the query string supplied is not part of the collection topic, we return 0 (IRRELEVANT)
        If the doc_id supplied is not part of the collection qrels, we return [0, 0, 0..]
        Else, we return the labels associated with doc_ids in qrels
        """
        assert isinstance(doc_ids, list)

        q_id = self.get_qid_from_query_string(query_string)
        if not q_id:
            # TODO: Is this assumption correct? Perhaps a different collection would have non-relevant as 1?
            return [self.IRRELEVANT] * len(doc_ids)

        all_relevant_docs = self.qrels.get(q_id)
        if all_relevant_docs is None:
            # No relevant documents for this topic. Unlikely
            return [self.IRRELEVANT] * len(doc_ids)

        relevances = [all_relevant_docs.get(doc_id, self.IRRELEVANT) for doc_id in doc_ids]

        return relevances

    def get_query_suggestions(self, query, num=10):
        """
        Returns topic titles that include the given query
        """
        titles = self.topics["title"]
        suggestions = []
        count = 0

        # TODO: Sloppy linear search. This may not scale if there are too many topics. Optimize later
        for q_id, title in titles.items():
            if query.lower() in title.lower():
                suggestions.append(title)
                count += 1
                if count >= num:
                    return suggestions

        return suggestions

    @staticmethod
    def validate_config(config):
        missing = [k for k in ["name"] + _data_keys if k not in config]
        if len(missing) > 0:
            raise RuntimeError(f"keys missing from collection config: {missing}")

        for key in _data_keys:
            if "path" not in config[key]:
                raise RuntimeError(f"missing path for key={key}")
            if "type" not in config[key]:
                config[key]["type"] = "trec"

    @staticmethod
    def normalize_paths(config, basepath):
        # join correctly handles the presence of a / in arguments
        for key in _data_keys:
            config[key]["path"] = os.path.join(basepath, os.path.expandvars(config[key]["path"]))

    @staticmethod
    def get_collection_from_index_path(index_path):
        """
        The index specified by index_path need NOT be the same as the index which was used to searcher the experiment\
        The user can choose any index from the dropdown in the UI
        This method tells you, based on the path, whether the index was made from a known collection (eg: robust04) or
        whether it's an index on something else (like a wikipedia dump)
        :param index_path: The path to the index
        :return: A collection class, or None
        """
        for name, collection in COLLECTIONS.items():
            if name in index_path:
                return collection

        return None


def to_trectxt(docno, txt):
    s = f"<DOC>\n<DOCNO> {docno} </DOCNO>\n"
    s += f"<TEXT>\n{txt}\n</TEXT>\n</DOC>\n"
    return s


def index_to_trec_docs(index_dir, output_dir, expected_doc_count):
    from jnius import autoclass

    JIndexUtils = autoclass("io.anserini.index.IndexUtils")
    index_utils = JIndexUtils(index_dir)

    docids = set()
    for i in range(expected_doc_count):
        try:
            docid = index_utils.convertLuceneDocidToDocid(i)
            docids.add(docid)
        except:
            # we reached the end?
            pass

    if len(docids) != expected_doc_count:
        raise ValueError(
            f"we expected to retrieve {expected_doc_count} documents from the index, but actually found {len(docids)}"
        )

    output_handles = [gzip.open(os.path.join(output_dir, f"{i}.gz"), "wt") for i in range(100, 200)]
    for docidx, docid in enumerate(sorted(docids)):
        txt = to_trectxt(docid, index_utils.getRawDocument(docid))
        handleidx = docidx % len(output_handles)
        print(txt, file=output_handles[handleidx])

    for handle in output_handles:
        handle.close()


COLLECTIONS = parse_collections()
