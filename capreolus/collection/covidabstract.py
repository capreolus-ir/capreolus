import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class CovidAbstract(IRDCollection):
    """TREC-COVID with only abstracts (no title or body)"""

    module_name = "covidabstract"
    ird_dataset_name = "cord19/trec-covid"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        return json.dumps({"id": doc.doc_id, "contents": doc.abstract})
