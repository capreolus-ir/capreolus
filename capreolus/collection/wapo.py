import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class WaPo(IRDCollection):
    """ TREC WashingtonPost v2 collection. See https://trec.nist.gov/data/wapost/ """

    module_name = "wapo"
    ird_dataset_name = "wapo/v2"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        content = " ".join((doc.title, doc.body))
        return json.dumps({"id": doc.doc_id, "contents": content})
