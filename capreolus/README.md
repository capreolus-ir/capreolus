# Overview

This repository illustrates the sacred-ingredient-based proposal for refactoring Capreolus' module API in order to (1) use a common API for all modules and (2) minimize the work performed by `pipeline.py` in preparation for supporting other pipelines, such as adding a third ranking stage or using non-pytorch-based methods as rerankers. My intention is add module-specific APIs, like a `crossvalidated_ranking` method for `Searcher` modules, but these have not yet been defined or implemented. As part of #2, modules now declare dependencies on other modules (e.g., a reranker requires an extractor, which requires an index to access document text) rather than requiring the pipeline to provide all dependencies.

# Module Dependencies and Config
The new `registry.py` defines a `ModuleBase` that all module classes should subclass (e.g. Collection, Reranker) and a `Dependency` class that modules can use to indicate which modules they depend on. For example, the `Searcher` class inherits from `ModuleBase` to define a `searcher` module type. The `SDM` class inherits from `Searcher` to provide a specific instantiation of a searcher. The SDM class declares a dependency on an `index` of type `anserini`: `dependencies = {"index": Dependency(module="index", name="anserini")}`. This dependency appears in the `dependencies` dict under the key `index` (called the "config key"), which means that its config will be accessible from `SDM` as `self.cfg['index']` and an `Index.Anserini` object created from this config will be available at `self.required_modules['index']`.

The Anserini Index depends on a `collection`. Rather than requiring a specific collection module (as was the case when `SDM` requested an Anserini index specifically), it uses the default of `name=None` to indicate that any collection module may be provided by the pipeline: `dependencies = {"collection": Dependency(module="collection")}`. The corresponding collection object will be accessible from the `Index` at `self.required_modules['collection']`.

Config options are now hierarchical in a way that reflects the dependency graph. For example, if the pipeline creates a `Searcher` module with the config key `searcher` (and this module follows the same dependency structure described above), the searcher's index can be set not to remove stopwords with the config option `searcher.index.keepstops=True`. Note that `searcher` and `index` here correspond to config keys -- `index` is the key used in `SDM.dependencies` and `searcher` is a top level module declared by the pipeline (`module_order` and `module_defaults`). Config keys do not necessarily match module names; for example, `SDM` would instead have `searcher.index1.keepstops` and `searcher.index2.keepstops` config options if it declared two index dependencies like this: `dependencies = {"index1": Dependency(module="index", name="anserini"), "index2": Dependency(module="index", name="anserini")}`.

# Tasks
The pipeline is defined by a Task module, which comes at the beginning of the command.
For example, `python -m capreolus.run rank.train` will run the train command in `task/rank.py`. Switching to `rerank.train` would have run the `train` command in `task/rerank.py`, which initializes a `Reranker` module in addition to those used by the Rank task.

# Skeleton Code
The `run.py` script instantiates modules in a way similar to capreolus' current approach.
That is, it:
- creates a configurable searcher, reranker, collection, and benchmark
- constructs an output path for the experiment (based on all config options)
- provides module instantiations for the four top level modules above and their dependencies
- DOES NOT use the module instantiations to train. This is one of the next steps, but first requires APIs to be defined for each module type. For example, `Index` modules should provide a method for retrieving documents from the index (which it can cache by using `self.get_cache_path()` in conjunctions with their document ids), `Reranker` modules should provide a `score` method, etc. This is the next step.

# Status
Assuming this seems like a reasonable path forward, the API next steps are described in the previous section (i.e., defining APIs for each module type).
In addition, the current implementation has several shortcomings that should be addressed:
- should dependencies be declared as classes rather than by name? This has the nice side effect of forcing a module to import all its dependencies, which means that `run.py` would no longer have to import all module base classes. (Importing is required so that the related name-to-class dicts are populated. The name mappings will still be required in order to provide command line options.)
- (minor) type casting of config options is not implemented
- (minor) sacred config files cannot be provided on the command line in place of config options. Handling this probably requires wrapping sacred's load_config_file and save_config_file to handle the `_name` renaming magic. (Module choices are provided as `collection=robust04` but get remapped to `collection._name=robust04` in order to make these options nicely with sacred.)

Try it out:
- `pip install -r requirements.txt`
- `python run.py`
- Use different settings for the searcher's and reranker's indexes: `python run.py rerank.describe with searcher.index.keepstops=True reranker.extractor.index.keepstops=False`
- Change the collection and note the change propagates to all collection dependencies (since `name=None`): `python run.py rerank.describe with collection=robust05`. (Also note that `collection=robust05` gets remapped to `collection._name=robust05`.)
