import importlib
import os
import sys

from collections import OrderedDict
from functools import partial
from glob import glob
from inspect import isclass

import sacred

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from capreolus.registry import PACKAGE_PATH, all_known_modules
from capreolus.task import Task
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Pipeline:
    """ Declare a pipeline consisting of one or more modules.

        The modules requested by the Task will be initialized following `Task.module_order`. If any module classes are not specified in `rewritten_args`, the defaults in `Task.module_defaults` will be used.

        Args:
            task_name (str): The name of a registered Task (e.g., rerank)
            rewritten_args (list): The list of command line arguments to pass to sacred (with the task name removed)
    """

    def __init__(self, task_name, rewritten_args, task_obj_passed=False):
        if task_obj_passed:
            self.task = task_name
        else:
            self.task = Task.plugins[task_name]
        self.rewritten_args = rewritten_args

        for module in self.task.module_order:
            # import the base module
            importlib.import_module(f"capreolus.{module}")

            # attempt to import any files in the module's subdirectory
            module_path = PACKAGE_PATH / module
            for fn in glob(os.path.join(module_path, "*.py")):
                modname = os.path.basename(fn)[:-3]
                if not (modname.startswith("__") or modname.startswith("flycheck_")):
                    importlib.import_module(f"capreolus.{module}.{modname}")

        # create a sacred experiment to attach config options, ingredients, etc. to
        self.ex = self._create_experiment(self.task.name)

        # add provided config_functions to experiment
        for config_function in self.task.config_functions:
            self.ex.config(config_function)

        # add provided ingredient config overrides
        for ingredient_path, k, v in self.task.config_overrides:
            self.ex.ingredient_lookup[ingredient_path].add_config({k: v})

        self.ex.default_command = self.task.default_command

    def _command_wrapper(self, _config, command_func):
        modules = self._create_modules(_config)
        return command_func(_config, modules)

    def _ingredient_command_wrapper(self, _config, command_func, path):
        modules = self._create_modules(_config)

        path_elements = path.split(".")
        current_module = modules[path_elements[0]]
        for path_element in path_elements[1:]:
            current_module = current_module.required_modules[path_element]

        return command_func(current_module)

    def run(self):
        """ Run the Pipeline described by this object.

            This involves first determining which config options to use (via sacred), given the defaults and any options specified by the user. The command to run is similarly determined. Next, the Task's command is called and passed the active config and active modules.
        """

        # TODO this is a hack to make Tasks show up in the sacred print msg. fix help messages to remove it.
        for command_name in reversed(sorted(self.task.plugins)):
            self.ex.commands[command_name] = self.ex.commands["print_config"]
            self.ex.commands.move_to_end(command_name, last=False)

        try:
            self.ex.run_commandline(argv=self.rewritten_args)
            del self.ex
        except:
            # delete experiment object so that we can create a new notebook pipeline without restarting kernel
            del self.ex
            raise

    def _create_module_ingredients(self, choices):
        """ Using any module `choices` and the module defaults in `self.task.module_defaults`, create ingredients for each module """

        ingredients = []
        ingredient_commands = []
        provided_modules = set()
        for module in self.task.module_order:
            module_name = choices.get(module, self.task.module_defaults.get(module))
            if module_name is None:
                raise Exception(
                    f"a {module} module was not declared in the module choices or pipeline defaults"
                )

            if module_name not in all_known_modules[module].plugins:
                raise KeyError(
                    f"could not find class for requested module {module}={module_name}"
                )

            module_cls = all_known_modules[module].plugins[module_name]
            module_ingredient, command_list = module_cls.resolve_dependencies(
                module, all_known_modules, provided_modules
            )

            ingredients.append(module_ingredient)
            ingredient_commands.extend(command_list)
            provided_modules.add(module)

        return ingredients, ingredient_commands

    def _extract_choices_from_argv(self, args):
        """ Given a list of command line arguments in `args`, return a dictionary specifying any module classes chosen """

        choices = {}
        # no config options were provided
        if "with" not in args:
            return choices

        # consider only text after the initial "with" statement, which indicates the beginning of config options
        args = args[args.index("with") + 1 :]

        for module in self.task.module_order:
            choices[module] = self._extract_module_choice_from_args(module, args)

        return {
            module: choice for module, choice in choices.items() if choice is not None
        }

    def _extract_module_choice_from_args(self, module, args):
        key = f"{module}="
        choice = None
        # if a key is repeated several times, we use the last value in order to match sacred's behavior
        for arg in args:
            if arg.startswith(key):
                choice = arg[len(key) :]
        return choice

    def _rewrite_argv_for_ingredients(self, args):
        """ Rewrite `args` so that module choices are converted to the correct notation for ingredients.
            e.g., option "collection=robust04" is rewritten to "collection._name=robust04"
            This is needed so that the former notation can be used by users. """

        # rewriting is not necessary if no config options were provided
        if "with" not in args:
            return args

        config_args = args[args.index("with") + 1 :]
        rewritten_args = args[: args.index("with") + 1]
        for arg in config_args:
            if "=" not in arg:
                # this is a filename
                print(
                    "WARNING: arguments provided in files may not be parsed correctly; _name handling is not implemented"
                )
                rewritten_args.append(arg)
            else:
                k, v = arg.split("=")
                for module in self.task.module_order:
                    if k == module:
                        arg = f"{module}._name={v}"
                rewritten_args.append(arg)

        return rewritten_args

    def _create_experiment(self, experiment_name, interactive=False):
        """ Create a sacred.Experiment containing config options for the chosen modules (and their dependencies) """

        chosen = self._extract_choices_from_argv(self.rewritten_args)
        self.rewritten_args = self._rewrite_argv_for_ingredients(self.rewritten_args)

        ingredients, ingredient_commands = self._create_module_ingredients(chosen)
        # for ingredient in ingredients:
        #    print_ingredient(ingredient)

        self.ex = sacred.Experiment(
            experiment_name, ingredients=ingredients, interactive=interactive
        )

        self.ex.ingredient_lookup = {}

        def _traverse_and_add_ingredients(children):
            for child in children:
                self.ex.ingredient_lookup[child.path] = child
                _traverse_and_add_ingredients(child.ingredients)

        _traverse_and_add_ingredients(self.ex.ingredients)

        # add task commands
        for command_name, command_func in self.task.commands.items():
            partial_func = partial(self._command_wrapper, command_func=command_func)
            partial_func.__name__ = command_name
            captured_func = self.ex.capture(partial_func)
            captured_func.unobserved = False  # TODO check
            self.ex.commands[command_name] = captured_func

        # add ingredient commands, which are subtly different from experiment-level commands (tasks).
        # We capture the function using the experiment config (as before), however,
        # we add the command name to the ingredient rather than the experiment so that sacred parses it correctly.
        for command_name, command_func, path, ingredient in ingredient_commands:
            partial_func = partial(
                self._ingredient_command_wrapper, command_func=command_func, path=path
            )
            partial_func.__name__ = command_name
            captured_func = self.ex.capture(partial_func)
            captured_func.unobserved = False  # TODO check
            ingredient.commands[command_name] = captured_func

        return self.ex

    def _create_modules(self, _config):
        """ Instantiate and return the chosen modules using the given config options """

        modules = OrderedDict()
        provided = {module: _config[module] for module in self.task.module_order}

        # fill in parts of the config that were provided at the top level
        for m in self.task.module_order:
            module_name = _config[m]["_name"]
            module_cls = all_known_modules[m].plugins[module_name]
            module_cls.add_missing_modules_to_config(
                _config[m], all_known_modules, provided
            )

        # instantiate models from the expanded config
        for m in self.task.module_order:
            module_name = _config[m]["_name"]
            module_cls = all_known_modules[m].plugins[module_name]
            module = module_cls.instantiate_from_config(_config[m], all_known_modules)

            modules[m] = module

        return modules


class Notebook:
    """ Construct a pipeline to be used interactively.

        The returned object contains `config` and `modules` attributes that are analogous to those used with a Task.

        The pipeline will consist of the modules in `pipeline_description` with the config options in `config`.
        Modules will be initialized in `module_order` if it is provided.
        If not, `Collection` modules are initialized first, followed by the remaining modules in alphabetical order.
        This is safe as long as Collection is the only required dependency. You will need to set `module_order` if not.

        Args:
            pipeline_description (dict or Task): either a dict describing the desired modules or a Task class whose environment should be created.
            If a Task is provided, it should be provided as a class rather than a class instance. e.g. `Notebook(task.rank.RankTask)`
            If a dict is provided, it must be in one of two valid formats:
                1. Simple format: `{module_type: module_class}`
                   For example, `{"collection": "robust04", "searcher": "BM25", "benchmark": "wsdm20demo"}`.
                   If this format is used, `module_type` is used as the module's name in the pipeline.
                   This means only one module of each module_type can be requested.
                2. [WIP] Full format: `{module_name: (module_type, module_class)}
                   For example, `{"collection1": ("collection", "robust04"), "collection2": ("collection", "antique")}`.
                   This format is more verbose, but it allows for full flexibility.
            config_string (str): Config string of the same form used on the command line. As with the CLI, default moduels may be overridden (e.g., `searcher=BM25Grid`) and module's config options may be changed (e.g., `searcher.b=0.5`).
            module_order: (list): Order in which to initialize the modules in a dict passed to `pipeline_description`. Ignored if a Task is passed. If None, a reasonable default will be used (see above).

        Returns:
            Notebook: an object with `config` and `modules` attributes.
    """

    def __init__(self, pipeline_description, config_string="", module_order=None):
        def interactive(config, modules):
            print("returning control to notebook")
            self.config = config
            self.modules = modules

            self.describe_pipeline = partial(
                Task.describe_pipeline, config=self.config, modules=self.modules
            )
            self.module_graph = partial(
                Task.module_graph, config=self.config, modules=self.modules
            )
            for command, func in self.task.commands.items():
                setattr(
                    self,
                    command,
                    partial(func, config=self.config, modules=self.modules),
                )

        if isinstance(pipeline_description, Task):
            raise RuntimeError(
                "Notebook requires a Task class, but you passed a Task object"
            )

        if isclass(pipeline_description) and issubclass(pipeline_description, Task):
            task = pipeline_description()
            task.commands.update({"interactive": interactive})
            task.default_command = interactive
            task.name = f"{task.name}.notebook"
        else:
            module_defaults = pipeline_description

            if not module_order:
                # move collection to the front, if present, then sort alphabetically.
                module_order = sorted(
                    module_defaults.keys(), key=lambda x: (x != "collection", x)
                )

            missing_modules = set(module_defaults.keys()) - set(module_order)
            if len(missing_modules) > 0:
                raise ValueError(
                    "When module_order is provided, it must contain every module in module_defaults, but these modules were missing: {missing_modules}"
                )

            self.config = None
            self.modules = None

            class NotebookTask(Task):
                def notebook_config():
                    seed = 123_456

                name = "notebook"
                module_order = None
                module_defaults = None
                config_functions = [notebook_config]
                config_overrides = []
                commands = {"interactive": interactive}
                default_command = interactive

            task = NotebookTask()
            task.module_order = module_order
            task.module_defaults = module_defaults

        config_args = config_string.split()
        if len(config_args) > 0 and config_args[0] != "with":
            config_args.insert(0, "with")
        rewritten_args = ["notebook", "interactive"] + config_args

        self.task = task
        pipeline = Pipeline(
            self.task, rewritten_args=rewritten_args, task_obj_passed=True
        )
        pipeline.run()
