import multiprocessing
import os
import random

import numpy as np
import sacred
import torch

from capreolus.reranker.reranker import Reranker
from capreolus.collection import COLLECTIONS
from capreolus.benchmark import Benchmark
from capreolus.index import Index
from capreolus.searcher import Searcher
from capreolus.utils.common import params_to_string, forced_types, get_default_cache_dir, get_default_results_dir
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus.utils.frozendict import FrozenDict
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

############################################################
# hack to allow config functions to return their default values
# (normally sacred does not allow config functions to return a value)
orig_dfb = sacred.config.config_scope.dedent_function_body


def _custom_config_dfb(*args, **kwargs):
    config_skip_return = "return locals().copy()  # ignored by sacred"
    src = orig_dfb(*args, **kwargs)
    filtered = [line for line in src.split("\n") if not line.strip() == config_skip_return]
    return "\n".join(filtered)


sacred.config.config_scope.dedent_function_body = _custom_config_dfb
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("CUDA_VISIBLE_DEVICES")
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("USER")
############################################################


modules = ("collection", "index", "searcher", "benchmark", "reranker")


def module_config():
    # default modules
    collection = "robust04"
    index = "anserini"
    searcher = "bm25"
    benchmark = "robust04.title.wsdm20demo"
    reranker = "PACRR"

    return locals().copy()  # ignored by sacred


# config options that shouldn't be automatically added to the path
# (e.g., they don't affect model training or they're manually included somewhere in the path)
def stateless_config():
    expid = "debug"  # experiment id/name
    predontrain = False
    fold = "s1"
    earlystopping = True
    return locals().copy()  # ignored by sacred


def pipeline_config():
    # not working / disabled
    # resume = False  # resume from last existing weights, if any exist #TODO make this work with epoch preds
    # saveall = True
    # selfprediction = False
    # uniformunk = True
    # datamode = "basic"

    maxdoclen = 800  # maximum document length (in number of terms after tokenization)
    maxqlen = 4  # maximum query length (in number of terms after tokenization)
    batch = 32  # batch size
    niters = 150  # number of iterations to train for
    itersize = 4096  # number of training instances in one iteration (epoch)
    gradacc = 1  # number of batches to accumulate over before updating weights
    lr = 0.001  # learning rate
    seed = 123_456  # random seed to use
    sample = "simple"
    softmaxloss = True  # True to use softmax loss (over pairs) or False to use hinge loss
    dataparallel = "none"

    if sample not in ["simple"]:
        raise RuntimeError(f"sample '{sample}' must be one of: simple")

    # sanity checks
    if niters <= 0:
        raise RuntimeError("niters must be > 0")

    if itersize < batch:
        raise RuntimeError("itersize must be >= batch")

    if niters < 1:
        raise RuntimeError("gradacc must be >= 1")

    return locals().copy()  # ignored by sacred


class Pipeline:
    def __init__(self, module_choices):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ex = sacred.Experiment("capreolus")
        self.ex = ex
        ex.path = "capreolus"
        ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

        self.module2cls = self.get_module_to_class(module_choices)

        # now the Modules to load have been determined, so we pass their configs to sacred and determine
        # which modules each config key should be associated with (based on first module to set it).
        # later modules can override keys, but the key remains associated with the initial module.
        # this is in order of lowest to highest precedence since later values override earlier ones with sacred
        self.parameters_to_module = self.get_parameters_to_module(ex)
        self.parameter_types = self.get_parameter_types(ex)

        self.parameters_to_module, self.parameter_types = self.get_parameters_to_module_for_missing_parameters(ex)
        self.parameters_to_module, self.parameter_types = self.get_parameters_to_module_for_feature_parameters(ex)
        self.module_to_parameters = self.get_module_to_parameters()
        self.check_for_invalid_keys()

    def check_for_invalid_keys(self):
        invalid_keys = []
        for k in self.parameters_to_module:
            if "_" in k or "-" in k or "," in k:
                invalid_keys.append(k)
        if len(invalid_keys) > 0:
            raise ValueError("config keys cannot contain '-' ',' or '_'\n\tinvalid keys: %s" % ", ".join(invalid_keys))

    def get_module_to_parameters(self):
        """
        Essentially the reverse of self.parameter_to_module. Associates with each module a list of parameter
        names that are valid for it
        """
        module_to_parameters = {module: [] for module in list(modules) + ["pipeline", "module", "stateless", "extractor"]}
        for k, module in self.parameters_to_module.items():
            module_to_parameters[module].append(k)

        return module_to_parameters

    def get_parameters_to_module_for_feature_parameters(self, ex):
        """
        Adds config related to the extractor associated with the supplied NIR Reranker to the parameter_to_module dict.
        Eg: See `EmbedText.config()`
        """
        self.parameter_types["extractor"] = str
        for feature_cls in self.module2cls["reranker"].EXTRACTORS:
            for k, v in ex.config(feature_cls.config)().items():
                if k in self.parameters_to_module:
                    raise RuntimeError(f"extractor {feature_cls} contains conflicting config key {k}")
                self.parameters_to_module[k] = "extractor"
                self.parameter_types[k] = forced_types.get(type(v), type(v))

        return self.parameters_to_module, self.parameter_types

    def get_parameters_to_module_for_missing_parameters(self, ex):
        """
        Not all parameters are supplied by the user. This method determines which parameters were not supplied by the
        user and plugs in sensible defaults for them
        """
        # config keys for each module
        for module in modules:
            if module == "collection":  # collections do not have their own configs
                continue
            for k, v in ex.config(self.module2cls[module].config)().items():
                if k not in self.parameters_to_module:
                    self.parameters_to_module[k] = module
                    self.parameter_types[k] = forced_types.get(type(v), type(v))

        return self.parameters_to_module, self.parameter_types

    def get_module_to_class(self, module_choices):
        # in order of highest to lowest precedence,
        # - determine the class of each module based on explicit (eg CLI param) choices or the default
        # - allow this module to override any module choices that were not explicit (i.e., set by user)
        module2cls = {}
        # collection < index < searcher < benchmark < Reranker
        module_loaders = {
            "collection": COLLECTIONS,
            "index": Index.ALL,
            "searcher": Searcher.ALL,
            "benchmark": Benchmark.ALL,
            "reranker": Reranker.ALL,
        }

        default_modules = module_config()
        for module in reversed(modules):
            # load user's choice or default module
            if module_choices.get(module, None) is None:
                module_choices[module] = default_modules[module]

            module2cls[module] = module_loaders[module][module_choices[module]]

            # collections do not have their own configs, so we stop early
            if module == "collection":
                continue

            # TODO: Is this required anymore? I don't think module_choices are used anywhere down the line
            # override any unset modules (which must have lower precedence)
            for k, v in module2cls[module].config().items():
                if k in modules and module_choices.get(k, None) is None:
                    logger.debug("%s config setting module %s = %s", module, k, v)
                    module_choices[k] = v

        return module2cls

    def get_parameter_types(self, ex):
        """
        For each config() parameter specified in the codebase for each module, deduce the correct type.
        Specifically, key_types[x] contains a function that we can call to cast the cmdline parameter to the correct type
        Eg: "none" should be casted to None
        """
        parameter_types = {}
        for k, v in ex.config(module_config)().items():
            parameter_types[k] = type("string")

        for k, v in ex.config(stateless_config)().items():
            parameter_types[k] = forced_types.get(type(v), type(v))

        parameter_types["pipeline"] = str
        for k, v in ex.config(pipeline_config)().items():
            parameter_types[k] = forced_types.get(type(v), type(v))

        return parameter_types

    def get_parameters_to_module(self, ex):
        """
        Creates a dict that group each of the supplied parameters to an umbrella "module"
        """
        parameter_to_module = {}
        for k, v in ex.config(module_config)().items():
            parameter_to_module[k] = "module"

        for k, v in ex.config(stateless_config)().items():
            parameter_to_module[k] = "stateless"

        for k, v in ex.config(pipeline_config)().items():
            parameter_to_module[k] = "pipeline"

        return parameter_to_module

    def get_paths(self, config):
        """
        Returns a dictionary of various paths
        :param config: A sacred config
        :return: A dict. Eg:
        {
            "collection_path": "path",
            "base_path": "path",
            "cache_path": "path",
            "index_path": "path",
            "run_path": "path",
            "model_path": "path"
        }
        """
        expid = config["expid"]
        collection_path = self.module2cls["collection"].basepath
        base_path = os.environ.get("CAPREOLUS_RESULTS", get_default_results_dir())
        cache_path = os.environ.get("CAPREOLUS_CACHE", get_default_cache_dir())
        index_key = os.path.join(cache_path, config["collection"], self.module_key("index"))
        index_path = os.path.join(index_key, "index")
        run_path = os.path.join(index_key, "searcher", self.module_key("searcher"))
        model_path = os.path.join(
            base_path,
            expid,
            config["collection"],
            self.module_key("index"),
            self.module_key("searcher"),
            self.module_key("benchmark"),
            self.module_key("pipeline"),
            self.module_key("reranker") + "_" + self.module_key("extractor"),
        )
        trained_weight_path = os.path.join(model_path, config["fold"], "weights", "dev")

        return {
            "collection_path": collection_path,
            "base_path": base_path,
            "cache_path": cache_path,
            "index_path": index_path,
            "index_key": index_key,
            "run_path": run_path,
            "model_path": model_path,
            "trained_weight_path": trained_weight_path,
        }

    def initialize(self, cfg):
        if hasattr(self, "cfg"):
            raise RuntimeError("Pipeline has already been initialized")

        cfg = {k: self.parameter_types[k](v) for k, v in cfg.items()}
        maxthreads = int(os.environ.get("CAPREOLUS_THREADS", multiprocessing.cpu_count()))
        if maxthreads <= 0:
            logger.warning("changing invalid maxthreads value of '%s' to 8", maxthreads)
            maxthreads = 8

        cfg["maxthreads"] = maxthreads
        self.cfg = FrozenDict(cfg)

        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])

        path_dict = self.get_paths(cfg)
        self.collection_path = path_dict["collection_path"]
        self.base_path = path_dict["base_path"]
        self.cache_path = path_dict["cache_path"]
        self.index_key = path_dict["index_key"]
        self.index_path = path_dict["index_path"]
        self.run_path = path_dict["run_path"]
        self.reranker_path = path_dict["model_path"]

        # attempt to download the collection if it is missing and a URL is available
        self.module2cls["collection"].download_if_missing(self.cache_path)

        if cfg["softmaxloss"]:
            self.lossf = pair_softmax_loss
        else:
            self.lossf = pair_hinge_loss

        # IMPORTANT - The order of initialization matters. Also, we need self.cfg to be present when control reaches here
        self.collection = self.module2cls["collection"]
        self.index = self.module2cls["index"](self.collection, self.index_path, self.index_key)
        self.searcher = self.module2cls["searcher"](self.index, self.collection, self.run_path, cfg)
        self.benchmark = self.module2cls["benchmark"](self.searcher, self.collection, cfg)
        self.benchmark.build()
        self.extractors = []
        self.initialize_extractors()
        self.reranker = self.module2cls["reranker"](
            self.extractors[0].embeddings, self.benchmark.reranking_runs[cfg["fold"]], cfg
        )
        self.reranker.build()
        self.reranker.to(self.device)
        self._anserini = None
        self.benchmark.set_extractor(self.extractors[0])

    def initialize_extractors(self):
        for cls in self.module2cls["reranker"].EXTRACTORS:
            cfg = {k: self.cfg[k] for k in cls.config()}
            extractor_cache_dir = self.extractor_cache(cls)
            extractor = cls(
                self.cache_path,
                extractor_cache_dir,
                self.cfg,
                benchmark=self.benchmark,
                collection=self.collection,
                index=self.index,
            )
            extractor.build_from_benchmark(**cfg)
            self.extractors.append(extractor)

    def extractor_cache(self, cls):
        cfg = {k: self.cfg[k] for k in cls.config()}
        cfg["extractor"] = cls.__name__
        feature_key = params_to_string("extractor", cfg, self.parameter_types)

        # HACK: set v=0 for keys that do not affect cache
        real_cfg = self.cfg
        self.cfg = {k: v for k, v in real_cfg.items()}
        for k in ["batch", "lr", "gradacc"]:
            self.cfg[k] = 0
        benchmark_key = self.module_key("benchmark")
        pipeline_key = self.module_key("pipeline")
        self.cfg = real_cfg

        s = os.path.join(
            self.cache_path,
            "features",
            self.cfg["collection"],
            self.module_key("index"),
            self.module_key("searcher"),
            benchmark_key,
            pipeline_key,
            self.cfg["fold"],
            feature_key,
        )
        return s

    def module_key(self, name):
        """
        Creates a string based on all the parameters and their values associated with a given module.
        This "key" can be used for caching, creating a directory structure e.t.c
        """
        compcfg = {k: self.cfg[k] for k in self.module_to_parameters[name]}
        # hack since the pipeline isn't an actual config option (and thus isn't included)
        if name == "pipeline" or name == "extractor":
            compcfg[name] = name
        else:
            compcfg[name] = self.cfg[name]
        return params_to_string(name, compcfg, self.parameter_types)


def cli_module_choice(argv, module):
    key = f"{module}="
    choice = None
    # if a key is repeated several times, we use the last value in order to match sacred's behavior
    for arg in argv:
        if arg.startswith(key):
            choice = arg[len(key) :]
    return choice
