import importlib
import multiprocessing
import os

from functools import partial
from pathlib import Path

# from types import MappingProxyType

import sacred

# mapping of module types (e.g., "collection") to the module base classes
all_known_modules = {}

PACKAGE_PATH = Path(os.path.dirname(__file__))
RESULTS_BASE_PATH = Path(os.environ.get("CAPREOLUS_RESULTS", os.path.expanduser("~/.capreolus/results/")))
MAX_THREADS = int(os.environ.get("CAPREOLUS_THREADS", multiprocessing.cpu_count()))
CACHE_BASE_PATH = Path(os.environ.get("CAPREOLUS_CACHE", os.path.expanduser("~/.capreolus/cache/")))


class Dependency:
    """ Represents a dependency on another module.

        If name is None, the dependency must be provided by the pipeline (i.e., in `provided_modules`).
        Otherwise, the module class corresponding to `name` will be used.

        If config_overrides is a dict, it will be used to override the dependency's default config options.
        Note that user may still override these options e.g. on the command line.
    """

    def __init__(self, module, name=None, config_overrides=None):
        importlib.import_module(f"capreolus.{module}")
        self.module = module
        self.name = name
        self.config_overrides = config_overrides

    def __str__(self):
        return f"<Dependency {self.module}={self.name} overrides={self.config_overrides}>"


class RegisterableModule(type):
    """ Metaclass indicating that the subclass is a Capreolus module.
        Modules receive a `self.plugins` dict mapping names to classes.
        Thie package's `all_known_modules` dict maps module names to module base classes. """

    def __init__(cls, name, parents, attrs):
        """ Metaclass used to automatically register module implementations """
        if not hasattr(cls, "plugins"):
            # true when module base class is declared (e.g., Collection)
            cls.plugins = {}
            all_known_modules[cls.module_type] = cls
        else:
            # class (Robust04) inheriting from the module base class (Collection)
            cls.register_plugin(cls)

    def register_plugin(cls, plugin):
        if cls.plugins.get(plugin.name, plugin) != plugin:
            print(f"WARNING: replacing entry {cls.plugins[plugin.name]} for {plugin.name} with {plugin}")
        cls.plugins[plugin.name] = plugin


class RegisterableMixIn:
    """ Class providing utility class methods for use with RegisterableModules. """

    @classmethod
    def add_missing_modules_to_config(cls, config, module_lookup, provided_modules):
        """ Expand config with the config entries for modules provided by the pipeline.
            Such config entries are not present for modules provided by the pipeline, such as Collection.
            config: config dict to expand
            module_lookup: a dict to use for module lookups, such as all_known_modules
            provided_modules: a dict mapping modules provided by the pipeline to their configs
        """

        for k, dependency in cls.dependencies.items():
            module, name = dependency.module, dependency.name
            # if this is a dependency that does not exist in config, it should be given in provided_modules.
            # this will be the case for shared dependencies declared at the top level, like collection.
            if k not in config:
                config[k] = provided_modules[k]
            else:
                module_lookup[module].plugins[name].add_missing_modules_to_config(config[k], module_lookup, provided_modules)

    @classmethod
    def instantiate_from_config(cls, config, module_lookup):
        """ Instantiate this module using a given config.
            config: config dict to use
            module_lookup: a dict to use for module lookups, such as all_known_modules
        """

        assert cls.plugins[config["_name"]] == cls, f'{config["_name"]} vs. {str(cls)}'

        self = cls(config)
        self.modules = {}
        for k, dependency in cls.dependencies.items():
            dependency_config = config[k]
            name = dependency_config["_name"]
            dependency_cls = module_lookup[dependency.module].plugins[name]
            self.modules[k] = dependency_cls.instantiate_from_config(dependency_config, module_lookup)

        return self

    @classmethod
    def _create_ingredient(cls, module, sub_ingredients, command_list):
        ingredient = sacred.Ingredient(module, ingredients=sub_ingredients.values())

        # create module config consisting of (1) the module class name and (2) its config options (from config())
        ingredient.add_config({"_name": cls.name})
        ingredient.config(cls.config)  # should be ingredient.config(cls.cfg)?

        # add ingredient's commands to the shared command_list
        for command_name, command_func in cls.commands.items():
            command_list.append((command_name, command_func, module, ingredient))

        # override sub_ingredients' configs
        for k, sub_ingredient in sub_ingredients.items():
            overrides = cls.dependencies[k].config_overrides
            if overrides:
                assert "_name" not in overrides, "cannot override _name"
                sub_ingredient.add_config(cls.dependencies[k].config_overrides)

        return ingredient

    @classmethod
    def resolve_dependencies(cls, name, module_lookup, provided_modules, command_list=None, prefix=None):
        """ Create an ingredient representing this module and its dependencies.
            Dependencies are resolved recursively with a depth-first search. 
            name: the config name of this ingredient (e.g. "collection", "searcher")
            module_lookup: a dict to use for module lookups, such as all_known_modules
            provided_modules: modules provided by the pipeline
            prefix: used to recursively set ingredient names; should be set to None on function call
        """

        if command_list is None:
            command_list = []

        if not prefix:
            prefix = []
        prefix = prefix.copy()
        prefix.append(name)
        path = ".".join(prefix)

        needed = {k: v for k, v in cls.dependencies.items() if k not in provided_modules}
        # do not include provided_modules modules because their configs will be added by add_missing_modules_to_config
        sub_ingredients = {
            k: module_lookup[dependency.module]
            .plugins[dependency.name]
            .resolve_dependencies(k, module_lookup, provided_modules, command_list, prefix=prefix)[0]
            for k, dependency in needed.items()
        }

        ingredient = cls._create_ingredient(path, sub_ingredients=sub_ingredients, command_list=command_list)
        return ingredient, command_list


class ModuleBase(RegisterableMixIn):
    """ Base class to be inherited by Capreolus module classes (e.g., Collection, Searcher) """

    def __init__(self, cfg):
        """ Use classmethod instantiate_from_config. """
        self.cfg = sacred.config.custom_containers.ReadOnlyDict(cfg)
        self.modules = {}

    # this module's dependencies: dict mapping config keys to Dependency objects
    dependencies = {}
    # this module's class methods that should be exposed as commands
    commands = {}
    cfg = None

    @staticmethod
    def config():
        pass

    def get_cache_path(self):
        """ Return a path encoding the module's config, which can be used for caching.
            The path is a function of the module's config and the configs of its dependencies.
        """
        return CACHE_BASE_PATH / self.get_module_path(include_provided=True)

    def get_module_path(self, include_provided=True):
        """ Return a path encoding the module's config, including its dependenceis """

        if include_provided:
            included_dependencies = [depname for depname in self.modules]
        else:
            included_dependencies = [depname for depname in self.modules if self.dependencies[depname].name is not None]

        if included_dependencies:
            prefix = os.path.join(
                *[self.modules[depname].get_module_path(include_provided=include_provided) for depname in included_dependencies]
            )
            return os.path.join(prefix, self._this_module_path_only())
        else:
            return self._this_module_path_only()

    def _this_module_path_only(self):
        """ Return a path encoding only the module's config (and not its dependencies' configs) """

        module_cfg = {k: v for k, v in self.cfg.items() if k not in self.dependencies}
        module_name_key = self.module_type + "-" + module_cfg.pop("_name")
        return "_".join([module_name_key] + [f"{k}-{v}" for k, v in sorted(module_cfg.items())])

    def __getitem__(self, key):
        return self.modules[key]


def print_ingredient(ingredient, prefix=""):
    childprefix = prefix + "  "
    print(prefix + ingredient.path)
    for child in ingredient.ingredients:
        print_ingredient(child, prefix=childprefix)


def print_module_graph(module, prefix=""):
    childprefix = prefix + "    "
    this = f"{module.module_type}={module.name}"
    print(prefix + this)
    for child in module.modules.values():
        print_module_graph(child, prefix=childprefix)
