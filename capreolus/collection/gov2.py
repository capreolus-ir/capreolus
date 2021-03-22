import os
import shutil
import tarfile

from capreolus import ConfigOption, constants, Dependency
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class Gov2(Collection):
    """ TREC Robust04 (TREC disks 4 and 5 without the Congressional Record documents) """

    module_name = "gov2"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_keys_not_in_path = ["path"]
    config_spec = [ConfigOption("path", "Aquaint-TREC-3-4", "path to corpus")]

    def download_if_missing(self):
        raise Exception("This should not have happened")
