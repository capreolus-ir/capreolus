import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class NYT(IRDCollection):
    """ New York Times collection. See https://catalog.ldc.upenn.edu/LDC2008T19 """

    module_name = "nyt"
    ird_dataset_name = "nyt"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        content = " ".join((doc.headline, doc.body))
        return json.dumps({"id": doc.doc_id, "contents": content})
