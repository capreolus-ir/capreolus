import os
import shutil
import tarfile

from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class Robust04(Collection):
    """ TREC Robust04 (TREC disks 4 and 5 without the Congressional Record documents) """

    module_name = "robust04"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_keys_not_in_path = ["path"]
    config_spec = [ConfigOption("path", "Aquaint-TREC-3-4", "path to corpus")]

    def download_if_missing(self):
        return self.download_index(
            url="https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz",
            sha256="dddb81f16d70ea6b9b0f94d6d6b888ed2ef827109a14ca21fd82b2acd6cbd450",
            index_directory_inside="index-robust04-20191213/",
            # this string should match how the index was built (i.e., Anserini, stopwords removed, Porter stemming)
            index_cache_path_string="index-anserini_indexstops-False_stemmer-porter",
            index_expected_document_count=528_030,
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
        logger.info("downloading index for missing collection %s to temporary file %s", self.module_name, archive_file)
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
