import importlib
import sys

from collections import OrderedDict
from functools import partial

import sacred

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from capreolus.registry import all_known_modules
from capreolus.task import Task

print('pipeline: ', __name__)


class Pipeline:
    def __init__(self, task_name, rewritten_args):
        """ Declare a pipeline consisting of one or more modules.
            The modules will be initialized in `module_order` with the default classes provided in `module_defaults`. """

        self.task = Task.plugins[task_name]
        self.rewritten_args = rewritten_args

        for module in self.task.module_order:
            importlib.import_module(module)

        # create a sacred experiment to attach config options, ingredients, etc. to
        self.ex = self.create_experiment(self.task.name)

        # add provided config_functions to experiment
        for config_function in self.task.config_functions:
            self.ex.config(config_function)

        # add provided ingredient config overrides
        for ingredient_path, k, v in self.task.config_overrides:
            self.ex.ingredient_lookup[ingredient_path].add_config({k: v})

        self.ex.default_command = self.task.default_command

    def _command_wrapper(self, _config, command_func):
        modules = self.create_modules(_config)
        return command_func(_config, modules)

    def _ingredient_command_wrapper(self, _config, command_func, path):
        modules = self.create_modules(_config)

        path_elements = path.split(".")
        current_module = modules[path_elements[0]]
        for path_element in path_elements[1:]:
            current_module = current_module.required_modules[path_element]

        return command_func(current_module)

    def run(self):
        # TODO this is a hack to make Tasks show up in the sacred print msg. fix help messages to remove it.
        for command_name in reversed(sorted(self.task.plugins)):
            self.ex.commands[command_name] = self.ex.commands["print_config"]
            self.ex.commands.move_to_end(command_name, last=False)

        self.ex.run_commandline(argv=self.rewritten_args)

    def _create_module_ingredients(self, choices):
        """ Using any module `choices` and the module defaults in `self.task.module_defaults`, create ingredients for each module """

        ingredients = []
        ingredient_commands = []
        provided_modules = set()
        for module in self.task.module_order:
            module_name = choices.get(module, self.task.module_defaults.get(module))
            if module_name is None:
                raise Exception(f"a {module} module was not declared in the module choices or pipeline defaults")

            module_cls = all_known_modules[module].plugins[module_name]
            module_ingredient, command_list = module_cls.resolve_dependencies(module, all_known_modules, provided_modules)

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

        return {module: choice for module, choice in choices.items() if choice is not None}

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
                print("WARNING: arguments provided in files may not be parsed correctly; _name handling is not implemented")
                rewritten_args.append(arg)
            else:
                k, v = arg.split("=")
                for module in self.task.module_order:
                    if k == module:
                        arg = f"{module}._name={v}"
                rewritten_args.append(arg)

        return rewritten_args

    def create_experiment(self, experiment_name, interactive=False):
        """ Create a sacred.Experiment containing config options for the chosen modules (and their dependencies) """

        chosen = self._extract_choices_from_argv(sys.argv)
        sys.argv = self._rewrite_argv_for_ingredients(sys.argv)

        ingredients, ingredient_commands = self._create_module_ingredients(chosen)
        # for ingredient in ingredients:
        #    print_ingredient(ingredient)

        self.ex = sacred.Experiment(experiment_name, ingredients=ingredients, interactive=interactive)

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
            partial_func = partial(self._ingredient_command_wrapper, command_func=command_func, path=path)
            partial_func.__name__ = command_name
            captured_func = self.ex.capture(partial_func)
            captured_func.unobserved = False  # TODO check
            ingredient.commands[command_name] = captured_func

        return self.ex

    def create_modules(self, _config):
        """ Instantiate and return the chosen modules using the given config options """

        modules = OrderedDict()
        provided = {module: _config[module] for module in self.task.module_order}

        # fill in parts of the config that were provided at the top level
        for m in self.task.module_order:
            module_name = _config[m]["_name"]
            module_cls = all_known_modules[m].plugins[module_name]
            module_cls.add_missing_modules_to_config(_config[m], all_known_modules, provided)

        # instantiate models from the expanded config
        for m in self.task.module_order:
            module_name = _config[m]["_name"]
            module_cls = all_known_modules[m].plugins[module_name]
            module = module_cls.instantiate_from_config(_config[m], all_known_modules)

            modules[m] = module

        return modules


class Notebook:
    def __init__(self, module_defaults, config_string="", module_order=None):
        """ Construct a pipeline consising of the modules in `module_defaults` and config options in `config`.
            Modules will be initialized in `module_order` if it is provided.
            If not, `Collection` modules are initialized first, followed by the remaining modules in alphabetical order.
            This is safe as long as Collection is the only required dependency. You will need to set `module_order` if not.
        """

        if not module_order:
            # move collection to the front, if present, then sort alphabetically.
            module_order = sorted(module_defaults.keys(), key=lambda x: (x != "collection", x))

        missing_modules = set(module_defaults.keys()) - set(module_order)
        if len(missing_modules) > 0:
            raise ValueError(
                "When module_order is provided, it must contain every module in module_defaults, but these modules were missing: {missing_modules}"
            )

        self.config = None
        self.modules = None

        def interactive(config, modules):
            print("returning control to notebook")
            self.config = config
            self.modules = modules

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

        NotebookTask.module_order = module_order
        NotebookTask.module_defaults = module_defaults

        config_args = config_string.split()
        if len(config_args) > 0 and config_args[0] != "with":
            config_args.insert(0, "with")
        rewritten_args = ["notebook", "interactive"] + config_args

        pipeline = Pipeline("notebook", rewritten_args=rewritten_args)
        pipeline.run()
