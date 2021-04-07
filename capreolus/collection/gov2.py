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
    collection_type = "TrecwebCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_keys_not_in_path = ["path"]
    config_spec = [ConfigOption("path", "/GW/NeuralIR/nobackup/GOV2/GOV2_data", "path to corpus")]

    def download_if_missing(self):
        raise Exception("This should not have happened")

    def _validate_document_path(self, path):
        """Validate that the document path appears to contain robust04's documents (Aquaint-TREC-3-4).

        Validation is performed by looking for four directories (case-insensitive): `FBIS`, `FR94`, `FT`, and `LATIMES`.
        These directories may either be at the root of `path` or they may be in `path/NEWS_data` (case-insensitive).

        Returns:
            True if the Aquaint-TREC-3-4 document directories are found or False if not
        """

        if not os.path.isdir(path):
            return False

        contents = {fn: fn for fn in os.listdir(path)}
        if "GX000" in contents and "GX272" in contents:
            return True

        return False


@Collection.register
class Gov2Passages(Collection):
    """ TREC Robust04 (TREC disks 4 and 5 without the Congressional Record documents) """

    module_name = "gov2passages"
    collection_type = "TrecwebCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_keys_not_in_path = ["path"]
    config_spec = [ConfigOption("path", "/GW/NeuralIR/nobackup/GOV2/GOV2_data", "path to corpus")]
    dependencies = [Dependency(key="task", module="task", name="gov2passages")]

    def download_if_missing(self):
        target_dir = os.path.join(self.task.get_cache_path(), "generated")
        if os.path.isdir(target_dir):
            return target_dir

        return self.download_index()

    def _validate_document_path(self, path):
        """
        Validate that the document path appears to contain robust04's documents (Aquaint-TREC-3-4).
        """

        if not os.path.isdir(path):
            return False

        contents = {fn.lower(): fn for fn in os.listdir(path)}

        if "generated" in contents:
            return True

        return False

    def download_index(self):
        self.task.generate()

        return os.path.join(self.task.get_cache_path(), "generated")


