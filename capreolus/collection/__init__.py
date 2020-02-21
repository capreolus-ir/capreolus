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
    _path = None

    def get_path_and_types(self):
        if not self.validate_document_path(self._path):
            self._path = self.find_document_path()

        return self._path, self.collection_type, self.generator_type

    def validate_document_path(self, path):
        """ Attempt to validate the document collection at `path`.

            By default, this will only check whether `path` exists. Subclasses should override
            `_validate_document_path(path)` with their own logic to perform more detailed checks.

            Returns:
                True if the path is valid following the logic described above, or False if it is not
         """

        if not (path and os.path.exists(path)):
            return False

        return self._validate_document_path(path)

    def _validate_document_path(self, path):
        """ Collection-specific logic for validating the document collection path. Subclasses should override this.

            Returns:
                this default method provided by Collection always returns true
         """

        return True

    def find_document_path(self):
        """ Find the location of this collection's documents (i.e., the raw document collection).

            We first check the collection's config for a path key. If found, `self.validate_document_path` checks
            whether the path is valid. Subclasses should override the private method `self._validate_document_path`
            with custom logic for performing checks further than existence of the directory. See `Robust04`.

            If a valid path was not found, call `download_if_missing`.
            Subclasses should override this method if downloading the needed documents is possible.

            If a valid document path cannot be found, an exception is thrown. 

            Returns:
                path to this collection's raw documents
        """

        # first, see if the path was provided as a config option
        if "path" in self.cfg and self.validate_document_path(self.cfg["path"]):
            return self.cfg["path"]

        # if not, see if the collection can be obtained through its download_if_missing method
        return self.download_if_missing()

    def download_if_missing(self):
        raise IOError(
            f"a download URL is not configured for collection={self.name} and the collection path does not exist; you must manually place the document collection at this path in order to use this collection"
        )


class Robust04(Collection):
    name = "robust04"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"
    config_keys_not_in_path = ["path"]

    @staticmethod
    def config():
        path = "Aquaint-TREC-3-4"

    def download_if_missing(self):
        return self.download_index(
            url="https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz",
            sha256="dddb81f16d70ea6b9b0f94d6d6b888ed2ef827109a14ca21fd82b2acd6cbd450",
            index_directory_inside="index-robust04-20191213/",
            # this string should match how the index was built (i.e., Anserini, stopwords removed, Porter stemming)
            index_cache_path_string="index-anserini_indexstops-False_stemmer-porter",
            index_expected_document_count=528030,
            cachedir=self.get_cache_path(),
        )

    def _validate_document_path(self, path):
        """ Validate that the document path appears to contain robust04's documents (Aquaint-TREC-3-4).

            Validation is performed by looking for four directories (case-insensitive): `FBIS`, `FR94`, `FT`, and `LATIMES`.
            These directories may either be at the root of `path` or they may be in `path/NEWS_data` (case-insensitive).

            Returns:
                True if the Aquaint-TREC-3-4 document directories are found or False if not
        """

        if not os.path.isdir(path):
            return False

        contents = {fn.lower(): fn for fn in os.listdir(path)}
        if "news_data" in contents:
            contents = {fn.lower(): fn for fn in os.listdir(os.path.join(path, contents["news_data"]))}

        if "fbis" in contents and "fr94" in contents and "ft" in contents and "latimes" in contents:
            return True

        return False

    def download_index(
        self, cachedir, url, sha256, index_directory_inside, index_cache_path_string, index_expected_document_count
    ):
        # Download the collection from URL and extract into a path in the cache directory.
        # To avoid re-downloading every call, we create an empty '/done' file in this directory on success.
        done_file = os.path.join(cachedir, "done")
        document_dir = os.path.join(cachedir, "documents")

        # already downloaded?
        if os.path.exists(done_file):
            return document_dir

        # 1. Download and extract Anserini index to a temporary location
        tmp_dir = os.path.join(cachedir, "tmp_download")
        archive_file = os.path.join(tmp_dir, "archive_file")
        os.makedirs(document_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info("downloading index for missing collection %s to temporary file %s", self.name, archive_file)
        download_file(url, archive_file, expected_hash=sha256)

        logger.info("extracting index to %s (before moving to correct cache path)", tmp_dir)
        with tarfile.open(archive_file) as tar:
            tar.extractall(path=tmp_dir)

        extracted_dir = os.path.join(tmp_dir, index_directory_inside)
        if not (os.path.exists(extracted_dir) and os.path.isdir(extracted_dir)):
            raise ValueError(f"could not find expected index directory {extracted_dir} in {tmp_dir}")

        # 2. Move index to its correct location in the cache
        index_dir = os.path.join(cachedir, index_cache_path_string, "index")
        if not os.path.exists(os.path.join("index_dir", "done")):
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
            shutil.move(extracted_dir, index_dir)

        # 3. Extract raw documents from the Anserini index to document_dir
        anserini_index_to_trec_docs(index_dir, document_dir, index_expected_document_count)

        # remove temporary files and create a /done we can use to verify extraction was successful
        shutil.rmtree(tmp_dir)
        with open(done_file, "wt") as outf:
            print("", file=outf)

        return document_dir


class DummyCollection(Collection):
    name = "dummy"
    _path = PACKAGE_PATH / "data" / "dummy" / "data"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class Robust05(Collection):
    name = "robust05"
    path = "missingpath"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"
