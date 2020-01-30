import os

from capreolus.pipeline import Pipeline
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


def search_files_or_folders_in_directory(path_to_top_dir, target_name):
    """
    Returns a list of files/folders in the given directory with the name specified in `target_name`.
    The list contains relative paths to the file.
    :param path_to_top_dir: a path to a director
    :param target_name: file name to search for. Plain string. Wildcards e.t.c are not supported yet
    :return: Relative path to files with the `target_file_name`
    """
    search_hits = []
    for root, dir_names, file_names in os.walk(path_to_top_dir):
        search_hits.extend([os.path.join(root, file_name) for file_name in file_names if file_name == target_name])
        search_hits.extend([os.path.join(root, dir_name) for dir_name in dir_names if dir_name == target_name])

    return search_hits


def copy_keys_to_each_other(dict_1, dict_2, val=None):
    """
    Make sure that all keys in one dict is in the other dict as well. Only the keys are copied over - the values are
    set as None
    eg:
    dict_1  = {
        "a": 1,
        "b": 2
    }

    dict_2 = {
        "a": 13,
        "c": 4
    }

    dict_1 after transformation = {
        "a" : 1,
        "b": 2,
        "c": None
    }

    dict_2 after transformation = {
        "a" : 13,
        "b": None,
        "c": 4
    }
    """

    for key in dict_1:
        if key not in dict_2:
            dict_2[key] = None

    for key in dict_2:
        if key not in dict_1:
            dict_1[key] = None

    return dict_1, dict_2
