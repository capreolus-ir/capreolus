import os

from . import Collection
from capreolus import constants, get_logger

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class DummyCollection(Collection):
    module_name = "dummy"
    _path = PACKAGE_PATH / "data" / "dummy" / "data"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"

    def _validate_document_path(self, path):
        """ Validate that the document path contains `dummy_trec_doc` """
        return "dummy_trec_doc" in os.listdir(path)
