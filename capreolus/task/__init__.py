import importlib
import json
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

    @staticmethod
    def describe_pipeline(config, modules, output_path=None):
        if not output_path:
            output_path = "[none defined]"

        print("module cache paths:")
        for module, obj in modules.items():
            print("  ", obj.get_cache_path())
        print("\n")

        print("--- module dependency graph ---")
        for module, obj in modules.items():
            obj.print_module_graph(prefix=" ")
        print("\n")

        print("\n------- config ----------------")
        print(json.dumps(config, indent=4))
        print("\n")

        print("\n\nresults path:", output_path)


# import all tasks so that the classes are always registered
pwd = os.path.dirname(__file__)
for fn in glob(os.path.join(pwd, "*.py")):
    modname = os.path.basename(fn)[:-3]
    if not (modname.startswith("__") or modname.startswith("flycheck_")):
        importlib.import_module(f"capreolus.task.{modname}")
