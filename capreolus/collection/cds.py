import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class CDS(IRDCollection):
    """ PMC collection subset used by TREC CDS 2016 """

    module_name = "cds"
    ird_dataset_name = "pmc/v2"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        content = " ".join((doc.title, doc.abstract, doc.body))
        return json.dumps({"id": doc.doc_id, "contents": content})
