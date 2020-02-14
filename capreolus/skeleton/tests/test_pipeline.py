from capreolus.pipeline import Pipeline


def test_extract_choices_from_argv():
    # manually assign these to avoid calling import_module
    pipeline = Pipeline([], {})
    pipeline.module_order = ["m1", "m2"]
    pipeline.module_defaults = {"m1": "cls1", "m2": "cls2"}

    arg_prefix = ["foo.py", "with"]

    choices = pipeline._extract_choices_from_argv(arg_prefix)
    assert choices == {}

    choices = pipeline._extract_choices_from_argv(arg_prefix + "m1=foo1 m2=foo2 m3=foo3".split())
    assert choices == {"m1": "foo1", "m2": "foo2"}


def test_rewrite_argv_for_ingredients():
    # manually assign these to avoid calling import_module
    pipeline = Pipeline([], {})
    pipeline.module_order = ["m1", "m2"]
    pipeline.module_defaults = {"m1": "cls1", "m2": "cls2"}

    arg_prefix = ["foo.py", "with"]

    rewritten_args = pipeline._rewrite_argv_for_ingredients(arg_prefix)
    assert rewritten_args == arg_prefix

    rewritten_args = pipeline._rewrite_argv_for_ingredients(arg_prefix + "m1=foo1 m2=foo2 m3=foo3".split())
    assert rewritten_args == arg_prefix + "m1._name=foo1 m2._name=foo2 m3=foo3".split()


def test_create_experiment():  # TODO
    pass
