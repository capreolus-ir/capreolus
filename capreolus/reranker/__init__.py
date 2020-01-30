import importlib
import os.path
from glob import glob

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

# import all model modules so that the model classes are registered
pwd = os.path.dirname(__file__)
logger.debug("checking for reranker to import in: %s", pwd)
for fn in glob(os.path.join(pwd, "*.py")):
    modname = os.path.basename(fn)[:-3]
    if not (modname.startswith("__") or modname.startswith("flycheck_")):
        importlib.import_module(f"capreolus.reranker.{modname}")
