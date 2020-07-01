import logging
import os
import sys

from docopt import docopt
from profane import DBManager, config_list_to_dict, constants

from capreolus.task import Task
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


def parse_task_string(s):
    fields = s.split(".")
    task = fields[0]
    task_cls = Task.lookup(task)

    if len(fields) == 2:
        cmd = fields[1]
    else:
        cmd = task_cls.default_command

    if not hasattr(task_cls, cmd):
        print("error: invalid command:", s)
        print(f"valid commands for task={task}: {sorted(task_cls.commands)}")
        sys.exit(2)

    return task, cmd


def prepare_task(fullcommand, config):
    taskstr, commandstr = parse_task_string(fullcommand)
    task = Task.create(taskstr, config)
    task_entry_function = getattr(task, commandstr)
    return task, task_entry_function


help = """
Usage:
    capreolus COMMAND [(with CONFIG...)] [options]
    capreolus help [COMMAND]
    capreolus (-h | --help)


    Options:
      -h --help                     Print this help message and exit.
      -l VALUE --loglevel=VALUE     Set the log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
      -p VALUE --priority=VALUE     Sets the priority for a queued up experiment. No effect without -q flag.
      -q --queue                    Queue this run, and do not start it.


    Arguments:
      PIPELINE  Name of pipeline to run, which consists of a Task and a command (see below for a list)
      CONFIG    Configuration assignments of the form foo.bar=17


    Tasks and their commands:
      rank.search             search a collection using queries from a benchmark
      rank.evaluate           evaluate the result of rank.search
      rank.searcheval         run rank.search followed by rank.evaluate

      rerank.train            run rank.search and train a model to rerank the results
      rerank.evaluate         evaluate the result of rerank.train
      rerank.traineval        run rerank.train followed by rerank.evaluate

      rererank.train          run rerank.train and train a (second) model to rerank the results
      rererank.evaluate       evaluate the result of rererank.train
      rererank.traineval      run rererank.train followed by rererank.evaluate

      tutorial.run            task from the "Getting Started" tutorial

    All tasks additionally support the following help commands: describe, print_config, print_pipeline
      e.g., capreolus rank.print_config with searcher=BM25
"""

if __name__ == "__main__":
    # hack to make docopt print full help message if no arguments are give
    if len(sys.argv) == 1:
        sys.argv.append("-h")

    arguments = docopt(help, version="TODO")

    if arguments["--loglevel"]:
        loglevel = arguments["--loglevel"].upper()
        valid_loglevels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

        if loglevel not in valid_loglevels:
            print("error: log level must be one of:", ", ".join(valid_loglevels))
            sys.exit(1)

        logging.getLogger("capreolus").setLevel(loglevel)

    # prepare task even if we're queueing, so that we validate the config
    config = config_list_to_dict(arguments["CONFIG"])
    task, task_entry_function = prepare_task(arguments["COMMAND"], config)

    if arguments["--queue"]:
        if not arguments["--priority"]:
            arguments["--priority"] = 0

        db = DBManager(os.environ.get("CAPREOLUS_DB"))
        db.queue_run(command=arguments["COMMAND"], config=config, priority=arguments["--priority"])
    else:
        logger.debug("starting command: %s", arguments["COMMAND"])
        logger.debug("config: %s", task.config)
        logger.debug("current constants: %s", constants)
        task_entry_function()
