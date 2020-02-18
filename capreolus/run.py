import sys
from capreolus.pipeline import Pipeline

# NEXT:
# - make pipeline use the function callback method (for commnd)
# - add commands for ingredients like index
# - add commands for train and evaluate to mirrow current setup
# sacred.arg_parser L41 -- use this to parse cmd line?


def parse_sacred_command(args, default="rerank"):
    """ Parse the command name from the command line args, which consists of a task and an optional command (e.g., rank.print_config)
        Return default if none was specified.
    """

    # remove command line flags
    args = [arg for arg in args if not arg.startswith("-")]

    # no arguments were provided after the script name
    if len(args) == 1:
        return default

    if "with" in args:
        # if "with" appears, the command should appear immediately before it
        index = args.index("with") - 1
    else:
        # there is no "with", so command must be the last argument
        index = len(args) - 1

    # if index points to the program name, no command was provided
    if index == 0:
        return default

    # index points to the command name
    return args[index]


if __name__ == "__main__":
    task_command_str = parse_sacred_command(sys.argv)
    task_command_path = task_command_str.split(".")
    task = task_command_path[0]
    command = ".".join(task_command_path[1:])

    rewritten_args = list(sys.argv)
    task_index = rewritten_args.index(task_command_str)
    rewritten_args[task_index] = command

    pipeline = Pipeline(task, rewritten_args)
    pipeline.run()
