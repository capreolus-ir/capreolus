"""
Tests related to modules present in capreolus/utils/
"""
import os
import pytest

from capreolus.pipeline import Pipeline
from capreolus.utils.common import padlist, Anserini, params_to_string, string_to_params


def test_padlist():
    a = [1, 2, 3]
    padded_a = padlist(a, 5)
    assert padded_a == [1, 2, 3, 0, 0]

    b = [1, 2, 3, 4, 5, 6]
    padded_b = padlist(b, 3)
    assert padded_b == [1, 2, 3]


def test_params_to_string():
    some_dict = {"name": "test", "hello": "world", "foo": "bar", "count": 3}

    param_types = {"name": type("test"), "hello": type("world"), "foo": type("bar"), "count": type(3)}

    key = params_to_string("name", some_dict, param_types)

    assert key == "test_count-3_foo-bar_hello-world"


def test_string_to_params():
    param_types = {"wololoo": type("test"), "hello": type("world"), "foo": type("bar"), "count": type(3)}
    params = string_to_params("wololoo", "test_count-3_foo-bar_hello-world", param_types)
    assert params == {"wololoo": "test", "count": 3, "foo": "bar", "hello": "world"}
