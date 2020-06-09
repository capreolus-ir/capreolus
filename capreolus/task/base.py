from profane import ModuleBase, Dependency, ConfigOption, constants, module_registry


class Task(ModuleBase):
    module_type = "task"
    commands = []
    help_commands = ["describe", "print_config", "print_paths", "print_pipeline"]
    default_command = "describe"
    requires_random_seed = True

    def print_config(self):
        print("Configuration:")
        self.print_module_config(prefix="  ")

    def print_paths(self):  # TODO
        pass

    def print_pipeline(self):
        print(f"Module graph:")
        self.print_module_graph(prefix="  ")

    def describe(self):
        self.print_pipeline()
        print("\n")
        self.print_config()

    def get_results_path(self):
        """ Return an absolute path that can be used for storing results.
            The path is a function of the module's config and the configs of its dependencies.
        """

        return constants["RESULTS_BASE_PATH"] / self.get_module_path()


@Task.register
class ModulesTask(Task):
    module_name = "modules"
    commands = ["list_modules"]
    default_command = "list_modules"

    def list_modules(self):
        for module_type in module_registry.get_module_types():
            print(f"module type={module_type}")

            for module_name in module_registry.get_module_names(module_type):
                print(f"       name={module_name}")
