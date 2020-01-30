import hashlib
import importlib
import os
import requests
import sys
from glob import glob

import jnius_config
from tqdm import tqdm
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

# type conversion functions to use for types where the default doesn't work (e.g., bool('false') == True)
forced_types = {type(True): lambda x: str(x).lower() == "true", type(None): lambda x: None if str(x).lower() == "none" else x}


class SoftFailure(Exception):
    pass


def padlist(list_to_pad, padlen, pad_token=0):
    """
    Pads a list with zeros
    """
    # TODO: Write a test for this
    padded_list = list_to_pad[:padlen] if len(list_to_pad) >= padlen else list_to_pad + [pad_token] * (padlen - len(list_to_pad))
    return padded_list


class Anserini:
    @classmethod
    def get_fat_jar(cls):
        basedir = get_capreolus_base_dir()
        paths = glob(os.path.join(basedir, 'capreolus', 'anserini-*-fatjar.jar'))

        latest = max(paths, key=os.path.getctime)
        return latest


def params_to_string(namekey, params, param_types, skip_check=False):
    params = {k: param_types[k](v) for k, v in params.items()}
    s = [params[namekey]]

    for k in sorted(params):
        # handled separately
        if k == namekey:
            continue
        # changing behavior so that default values are also shown
        # if k not in always_params and k in param_defaults and params[k] == param_defaults[k]:
        #    continue
        s.append("%s-%s" % (k, params[k]))

    s = "_".join(s)

    if not skip_check:
        d = string_to_params(namekey, s, param_types, skip_check=True)
        if params != d:
            for k, v in d.items():
                if k not in params or params.get(k) != v:
                    logger.error(
                        "params_to_string mismatch: %s k=%s vs. k=%s ; types: %s vs. %s",
                        k,
                        params.get(k),
                        d[k],
                        type(params.get(k)),
                        type(d[k]),
                    )
            for k, v in params.items():
                if k not in d or d.get(k) != v:
                    logger.error(
                        "params_to_string mismatch: %s k=%s vs. k=%s ; types: %s vs. %s",
                        k,
                        params[k],
                        d.get(k),
                        type(params[k]),
                        type(d.get(k)),
                    )
            logger.error("sorted dict: %s", sorted(d.items()))
            logger.error("mismatch: string: %s", s)
            raise RuntimeError("error serializing params when running string_to_params(s, True)")
    return s


def string_to_params(namekey, s, param_types, skip_check=False):
    fields = s.split("_")
    params = fields[1:]

    out = {}
    out[namekey] = fields[0]

    for pstr in params:
        k, v = pstr.split("-", 1)
        out[k] = param_types[k](v)

    if not skip_check:
        assert s == params_to_string(namekey, out, param_types, skip_check=True), "asymmetric string_to_params on string: %s" % s
    return out


def register_component_module(cls, subcls):
    if not hasattr(subcls, "name"):
        raise RuntimeError(f"missing name attribute: {subcls}")

    name = subcls.name
    # ensure we don't have two modules containing subclasses with the same name
    if name in cls.ALL and cls.ALL[name] != subcls:
        raise RuntimeError(f"encountered two benchmarks with the same name: {name}")

    cls.ALL[name] = subcls
    return subcls


def get_capreolus_base_dir():
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    return "{0}/../../".format(curr_file_dir)


def import_component_modules(name):
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = "{0}/..".format(curr_file_dir)
    moddir = name
    logger.debug(f"loading {name} modules from {moddir}/")
    for fn in glob(os.path.join("{0}/{1}".format(root_dir, moddir), "*.py")):
        modname = os.path.basename(fn)[:-3]
        if not (modname.startswith("__") or modname.startswith("flycheck_")):
            importlib.import_module(f"capreolus.{moddir}.{modname}")


def get_default_cache_dir():
    return "{0}/cache".format(get_capreolus_base_dir())


def get_default_results_dir():
    return "{0}/results".format(get_capreolus_base_dir())


def get_crawl_collection_script():
    return "{0}/capreolus/crawl_collection.py".format(get_capreolus_base_dir())


def args_to_key(name, params):
    """
    Given a list of arguments (params), converts it into a string that can be used as a key for caching
    """
    params["name"] = name
    types = {k: forced_types.get(type(v), type(v)) for k, v in params.items()}
    key = params_to_string("name", params, types)
    return key


def download_file(url, outfn, expected_hash=None):
    """ Download url to the file outfn. If expected_hash is provided, use it to both verify the file was downloaded
        correctly, and to avoid re-downloading an existing file with a matching hash.
    """

    if expected_hash and os.path.exists(outfn):
        found_hash = hash_file(outfn)

        if found_hash == expected_hash:
            return

    head = requests.head(url)
    size = int(head.headers.get("content-length", 0))

    with open(outfn, "wb") as outf:
        r = requests.get(url, stream=True)
        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"downloading {url}", miniters=1) as pbar:
            for chunk in r.iter_content(32 * 1024):
                outf.write(chunk)
                pbar.update(len(chunk))

    if not expected_hash:
        return

    found_hash = hash_file(outfn)
    if found_hash != expected_hash:
        raise IOError("expected file {outfn} downloaded from {url} to have SHA256 hash {expected_hash} but got {found_hash}")


def hash_file(fn):
    """ Compute a SHA-256 hash for the file fn and return a hexdigest of the hash """
    sha = hashlib.sha256()

    with open(fn, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha.update(data)

    return sha.hexdigest()
