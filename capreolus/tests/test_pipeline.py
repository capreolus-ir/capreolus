import pytest

from collections import namedtuple
from capreolus.pipeline import Pipeline, Notebook


class DummyPipeline(Pipeline):
    def __init__(self, task):
        self.task = task


def test_extract_choices_from_argv():
    # manually assign these to avoid calling import_module
    pipeline = DummyPipeline(namedtuple("task", "module_order")(["m1", "m2"]))

    arg_prefix = ["foo.py", "with"]

    choices = pipeline._extract_choices_from_argv(arg_prefix)
    assert choices == {}

    choices = pipeline._extract_choices_from_argv(arg_prefix + "m1=foo1 m2=foo2 m3=foo3".split())
    assert choices == {"m1": "foo1", "m2": "foo2"}


def test_rewrite_argv_for_ingredients():
    # manually assign these to avoid calling import_module
    pipeline = DummyPipeline(namedtuple("task", "module_order")(["m1", "m2"]))

    arg_prefix = ["foo.py", "with"]

    rewritten_args = pipeline._rewrite_argv_for_ingredients(arg_prefix)
    assert rewritten_args == arg_prefix

    rewritten_args = pipeline._rewrite_argv_for_ingredients(arg_prefix + "m1=foo1 m2=foo2 m3=foo3".split())
    assert rewritten_args == arg_prefix + "m1._name=foo1 m2._name=foo2 m3=foo3".split()


# we don't currently test other Pipeline functions, which are tightly coupled,
# but we do test that Tasks are created as we expect
def test_simple_task_construction():
    module_defaults = {"searcher": "BM25", "collection": "robust04", "benchmark": "wsdm20demo"}
    nb = Notebook(module_defaults)

    for module_type, module_class in module_defaults.items():
        assert module_type in nb.config
        assert nb.config[module_type]["_name"] == module_class

        assert module_type in nb.modules
        assert nb.modules[module_type].name == nb.config[module_type]["_name"]


def test_task_construction_with_config():
    module_defaults = {"searcher": "BM25", "collection": "robust04", "benchmark": "wsdm20demo"}
    config_string = "searcher=BM25Grid"
    nb = Notebook(module_defaults, config_string=config_string)

    assert nb.config["searcher"]["_name"] == "BM25Grid"
    assert "k1" not in nb.config["searcher"]  # only present for BM25
    assert "bmax" in nb.config["searcher"]  # only present for BM25Grid

    assert "searcher" in nb.modules
    assert nb.modules["searcher"].name == nb.config["searcher"]["_name"]


def test_task_construction_failure_on_bad_config():
    module_defaults = {"searcher": "BM25", "collection": "robust04", "benchmark": "wsdm20demo"}

    # sacred calls exit(1) when the config cannot be parsed correctly
    with pytest.raises(SystemExit):
        nb = Notebook(module_defaults, config_string="unknownoption=wasadded")

    # _create_module_ingredients fails if a module cannot be found
    with pytest.raises(KeyError):
        nb = Notebook(module_defaults, config_string="searcher=nosuchsearchermodule")

    # and still fails if the module and module class are valid but the combination is not
    with pytest.raises(KeyError):
        nb = Notebook(module_defaults, config_string="searcher=robust04")
