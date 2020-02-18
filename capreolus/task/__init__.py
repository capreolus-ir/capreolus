import importlib
import os
from glob import glob

from capreolus.registry import RegisterableModule


class Task(metaclass=RegisterableModule):
    module_type = "task"

    name = None
    module_order = []
    module_defaults = {}
    config_functions = []
    config_overrides = []


# import all tasks so that the classes are always registered
pwd = os.path.dirname(__file__)
for fn in glob(os.path.join(pwd, "*.py")):
    modname = os.path.basename(fn)[:-3]
    if not (modname.startswith("__") or modname.startswith("flycheck_")):
        importlib.import_module(f"capreolus.task.{modname}")
