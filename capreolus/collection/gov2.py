from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class Gov2(IRDCollection):
    """ GOV2 collection from http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html """

    module_name = "gov2"
    ird_dataset_name = "gov2"
    collection_type = "TrecwebCollection"
