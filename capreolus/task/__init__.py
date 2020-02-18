import importlib
import json
import os
from glob import glob

from capreolus.registry import RegisterableModule, print_module_graph


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

        print("cache paths:")
        for module, obj in modules.items():
            print("  ", obj.get_cache_path())
        print("")

        print("--- module dependency graph ---")
        for module, obj in modules.items():
            print_module_graph(obj, prefix=" ")
        print("-------------------------------")

        print("\n------- config ----------------")
        print(json.dumps(config, indent=4))
        print("-------------------------------")

        print("\n\nresults path:", output_path)


# import all tasks so that the classes are always registered
pwd = os.path.dirname(__file__)
for fn in glob(os.path.join(pwd, "*.py")):
    modname = os.path.basename(fn)[:-3]
    if not (modname.startswith("__") or modname.startswith("flycheck_")):
        importlib.import_module(f"capreolus.task.{modname}")
