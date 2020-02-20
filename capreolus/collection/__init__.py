import os
import shutil
import tarfile

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs

logger = get_logger(__name__)


class Collection(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "collection"
    is_large_collection = False

    def get_path_and_types(self):
        if not os.path.exists(self.path):
            self.download_if_missing()

        return self.path, self.collection_type, self.generator_type

    def download_if_missing(self):
        raise IOError(
            f"a download URL is not configured for collection={self.name} and the collection path {self.path} does not exist; you must manually place the document collection at this path in order to use this collection"
        )


class Robust04(Collection):

    name = "robust04"
    path = "/home/andrew/Aquaint-TREC-3-4"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    def download_if_missing(self):
        if os.path.exists(self.path):
            return
        else:
            self.download_index(
                url="https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz",
                sha256="dddb81f16d70ea6b9b0f94d6d6b888ed2ef827109a14ca21fd82b2acd6cbd450",
                index_directory_inside="index-robust04-20191213/",
                # this string should match how the index was built (i.e., Anserini with stopwords removed and Porter stemming)
                index_cache_path_string="index-anserini_indexstops-False_stemmer-porter",
                index_expected_document_count=528030,
                cachedir=self.get_cache_path(),
            )

    def download_index(
        self, cachedir, url, sha256, index_directory_inside, index_cache_path_string, index_expected_document_count
    ):
        # Download the collection from URL and extract into a path in the cache directory.
        # To avoid re-downloading every call, we create an empty '/done' file in this directory on success.
        done_file = os.path.join(cachedir, "done")
        document_dir = os.path.join(cachedir, "documents")
        # set self.path to ensure calls to get_path_and_types return the correct path (rather than returning the default)
        self.path = document_dir

        # already downloaded?
        if os.path.exists(done_file):
            return True

        # 1. Download and extract Anserini index to a temporary location
        tmp_dir = os.path.join(cachedir, "tmp")
        archive_file = os.path.join(tmp_dir, "archive_file")
        os.makedirs(document_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info("downloading index for missing collection %s to temporary file %s", self.name, archive_file)
        download_file(url, archive_file, expected_hash=sha256)

        logger.debug("extracting to %s", tmp_dir)
        with tarfile.open(archive_file) as tar:
            tar.extractall(path=tmp_dir)

        extracted_dir = os.path.join(tmp_dir, index_directory_inside)
        if not (os.path.exists(extracted_dir) and os.path.isdir(extracted_dir)):
            raise ValueError(f"could not find expected index directory {extracted_dir} in {tmp_dir}")

        # 2. Move index to its correct location in the cache
        index_dir = os.path.join(cachedir, self.name, index_cache_path_string, "index")
        shutil.move(extracted_dir, index_dir)

        # 3. Extract raw documents from the Anserini index to document_dir
        anserini_index_to_trec_docs(index_dir, document_dir, index_expected_document_count)

        # remove temporary file and create a /done we can use to verify extraction was successful
        os.remove(archive_file)
        with open(done_file, "wt") as outf:
            print("", file=outf)


class DummyCollection(Collection):
    name = "dummy"
    path = PACKAGE_PATH / "data" / "dummy" / "data"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class Robust05(Collection):
    name = "robust05"
    path = "missingpath"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"
