import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)


@Collection.register
class MSMarcoDoc(IRDCollection):
    module_name = "msmarcodoc"
    ird_dataset_name = "msmarco-document"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        content = " ".join((doc.title, doc.body))
        return json.dumps({"id": doc.doc_id, "contents": content})
