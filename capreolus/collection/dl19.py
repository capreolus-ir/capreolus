from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)


@Collection.register
class DL19(IRDCollection):
    """TREC-DL-2019 collection from https://ir-datasets.com/msmarco-document.html#msmarco-document/trec-dl-2019"""

    module_name = "dl19"
    ird_dataset_name = "msmarco-document/trec-dl-2019"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
