import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class Highwire(IRDCollection):
    """Highire collection used by TREC Genomics 2006 and 2007"""

    module_name = "highwire"
    ird_dataset_name = "highwire"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        content = " ".join((span.text for span in doc.spans))
        return json.dumps({"id": doc.doc_id, "contents": content})
