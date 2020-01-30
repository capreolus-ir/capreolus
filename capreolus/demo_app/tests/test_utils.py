import os
import shutil
import json

from capreolus.demo_app.utils import search_files_or_folders_in_directory, copy_keys_to_each_other


def create_dummy_configs(tmpdir):
    config_1 = {"name": "test_config_1", "model": "some_model_name"}
    config_2 = {"name": "test_config_2", "model": "KNRM"}

    # write these configs to files
    tmpdir.mkdir("dummy_dir")
    tmpdir.mkdir("dummy_dir/nested_dummy_dir_1/")
    tmpdir.mkdir("dummy_dir/nested_dummy_dir_2/")

    tmpdir.join("dummy_dir/nested_dummy_dir_1/config.json").write(json.dumps(config_1))
    tmpdir.join("dummy_dir/nested_dummy_dir_2/config.json").write(json.dumps(config_2))


def test_search_files_in_directory(tmpdir):
    create_dummy_configs(tmpdir)
    files = search_files_or_folders_in_directory(tmpdir.strpath, "config.json")
    assert set(files) == {
        tmpdir.strpath + "/dummy_dir/nested_dummy_dir_1/config.json",
        tmpdir.strpath + "/dummy_dir/nested_dummy_dir_2/config.json",
    }


def test_set_different_keys():
    dict_1 = {"a": "hello", "b": "world"}

    dict_2 = {"a": "foo", "c": ["bar"]}

    dict_1, dict_2 = copy_keys_to_each_other(dict_1, dict_2)
    assert dict_1 == {"a": "hello", "b": "world", "c": None}

    assert dict_2 == {"a": "foo", "b": None, "c": ["bar"]}
