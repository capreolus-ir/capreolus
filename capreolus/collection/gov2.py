import os
import shutil
import tarfile

from capreolus import ConfigOption, constants, Dependency
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs

from . import Collection, IRDCollection

logger = get_logger(__name__)


@Collection.register
class Gov2(IRDCollection):
    """GOV2 collection from http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html"""

    module_name = "gov2"
    ird_dataset_name = "gov2"
    collection_type = "TrecwebCollection"


@Collection.register
class Gov2Passages(Collection):
    """ TREC Robust04 (TREC disks 4 and 5 without the Congressional Record documents) """

    module_name = "gov2passages"
    collection_type = "TrecCollection"
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