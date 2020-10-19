# Getting Started

- Requirements: Python 3.7+, a Python environment you can install packages in (e.g., a [Conda environment](https://gist.github.com/andrewyates/970c570411c4a36785f6c0e9362eb1eb)), and Java 11. See the [detailed installation instructions](installation) for help with these.
- Install: `pip install capreolus`

```eval_rst
.. note:: Results and cached objects are stored in ``~/.capreolus/results/`` and ``~/.capreolus/cache/`` by default. Set the ``CAPREOLUS_RESULTS`` and ``CAPREOLUS_CACHE`` environment variables to change these locations. For example: ``export CAPREOLUS_CACHE=/data/capreolus/cache``
```

## Command Line Interface

Use the `RankTask` pipeline to rank documents using a `Searcher` on an [Anserini](https://anserini.io) `Index` built on [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/), which contains biomedical documents and queries. NFCorpus was published by Boteva et al. in ECIR 2016. This dataset is publicly available and will be automatically downloaded by Capreolus.

```bash
$ capreolus rank.searcheval with benchmark.name=nf \
  searcher.name=BM25 searcher.index.stemmer=porter searcher.b=0.8
```

The `searcheval` command instructs `RankTask` to query NFCorpus and evaluate the Searcher's performance on NFCorpus' test queries. The command will output results like this:
```bash
INFO - capreolus.task.rank.evaluate - rank: fold=s1 best run: ...searcher-BM25_b-0.8_fields-title_hits-1000_k1-0.9/task-rank_filter-False/searcher
INFO - capreolus.task.rank.evaluate - rank: cross-validated results when optimizing for 'map':
INFO - capreolus.task.rank.evaluate -             map: 0.1520
INFO - capreolus.task.rank.evaluate -     ndcg_cut_10: 0.3247
...
```

These results are comparable with the *all titles* results in the [NFCorpus paper](https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf), which reports a MAP of 0.1251 for BM25 (Table 2). The Benchmark's ``fields`` config option can be used to issue other types of queries as well (e.g., ``benchmark.fields=all_fields``).

```eval_rst
.. important:: Capreolus Benchmarks define *folds* to use; each fold specifies training, dev (validation), and test queries.
          Tasks respect these folds when calculating metrics.
          NFCorpus defines a fixed test set, which corresponds to having a *single fold* in Capreolus.
          When running a benchmark that uses multiple folds with cross-validation, like *robust04*, the results reported are averaged over the benchmark's test sets.
```

## Python API

Let's run the same pipeline using the Python API:
```python
from capreolus.task import RankTask

task = RankTask({'searcher': {'name': 'BM25', 'index': {'stemmer': 'porter'}, 'b': '0.8'},
                 'benchmark': {'name': 'nf'}})
task.searcheval()
```

```eval_rst
.. note:: The ``capreolus.parse_config_string`` convenience method can transform a config string like ``searcher.name=BM25 benchmark.name=nf`` into a config dict as shown above.
```

<img style="float: right" src="_static/ranktask.png">


<p style="text-align: justify">
Capreolus pipelines are composed of self-contained modules corresponding to "IR primitives", which can also be used individually. Each module declares any module dependencies it needs to perform its function. The pipeline itself, which can be viewed as a dependency graph, is represented by a <code class="docutils literal notranslate"><span class="pre">Task</span></code> module.
</p>

<p style="text-align: justify">
<code class="docutils literal notranslate"><span class="pre">RankTask</span></code> declares dependencies on a <code class="docutils literal notranslate"><span class="pre">Searcher</span></code> module and a <code class="docutils literal notranslate"><span class="pre">Benchmark</span></code> module, which it uses to query a document collection and to obtain experimental data (i.e., topics, relevance judgments, and folds), respectively. The <code class="docutils literal notranslate"><span class="pre">Searcher</span></code> depends on an <code class="docutils literal notranslate"><span class="pre">Index</span></code>. Both the <code class="docutils literal notranslate"><span class="pre">Index</span></code> and <code class="docutils literal notranslate"><span class="pre">Benchmark</span></code> depend on a <code class="docutils literal notranslate"><span class="pre">Collection</span></code>. In this example, <code class="docutils literal notranslate"><span class="pre">RankTask</span></code> requires that the same <code class="docutils literal notranslate"><span class="pre">Collection</span></code> be provided to both.
</p>

```python
from capreolus import Benchmark, Collection, Index, Searcher
```

Let's construct this graph one module at a time.
```python
# Previously, the Benchmark specified a dependency on the 'nf' collection specifically.
# Now we create this Collection directly.
>>> collection = Collection.create("nf")
>>> collection.get_path_and_types()
    ("/path/to/collection-nf/documents", "TrecCollection", "DefaultLuceneDocumentGenerator")
# Next, create a Benchmark and pass it the collection object directly.
# This is an alternative to automatically creating the collection as a dependency.
>>> benchmark = Benchmark.create("nf", provide={'collection': collection})
>>> benchmark.topics["title"]
    {'56': 'foods for glaucoma', '68': 'what is actually in chicken nuggets', ... }
```

Next, we can build `Index` and `Searcher`. These module types do more than just pointing to data.
```python
>>> index = Index.create("anserini", {"stemmer": "porter"}, provide={"collection": collection})
>>> index.create_index()  # returns immediately if the index already exists
>>> index.get_df("foods")
0
>>> index.get_df("food")
1011
# Next, a Searcher to query the index
>>> searcher = Searcher.create("BM25", {"hits": 3}, provide={"index": index})
>>> searcher.query("foods")
OrderedDict([('MED-1761', 1.213), 
             ('MED-2742', 1.212),
             ('MED-1046', 1.2058)])
```

Finally, we can emulate the `RankTask.search()` method we called earlier:
```python
>>> results = {}
>>> for qid, topic in benchmark.topics['title'].items():
        results[qid] = searcher.query(topic)
```
To get metrics, we could then pass `results` to `capreolus.evaluator.eval_runs()`:
```eval_rst
.. autoapifunction:: capreolus.evaluator.eval_runs
   :noindex:
```


## Creating New Modules

Capreolus modules implement the Capreolus module API plus an API specific to the module type.
The module API consists of four attributes:
- `module_type`: a string indicating the module's type, like "index" or "benchmark"
- `module_name`: a string indicating the module's name, like "anserini" or "nf"
- `config_spec`: a list of `ConfigOption` objects. For example, `[ConfigOption("stemmer", default_value="none", description="stemmer to use")]`
- `dependencies` a list of `Dependency` objects. For example, `[Dependency(key="collection", module="collection", name="nf")]`

When the module is created, any dependencies that are not explicitly passed with `provide={key: object}` are automatically created. The module's config options in `config_spec` and those of its dependencies are exposed as Capreolus configuration options.


### Task API

The `Task` module API specifies two additional class attributes: `commands` and `default_command`. These specify the functions that should serve as the Task's entrypoints and the default entrypoint, respectively.

Let's create a new task that mirrors the graph we constructed manually, except with two separate `Searcher` objects. We'll save the results from both searchers and measure their effectiveness on the validation queries to decide which searcher to report test set results on.

```python
from capreolus import evaluator, get_logger, Dependency, ConfigOption
from capreolus.task import Task

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Task.register
class TutorialTask(Task):
    module_name = "tutorial"
    config_spec = [ConfigOption("optimize", "map", "metric to maximize on the validation set")]
    dependencies = [
        Dependency(
            key="benchmark", module="benchmark", name="nf", provide_this=True, provide_children=["collection"]
        ),
        Dependency(key="searcher1", module="searcher", name="BM25RM3"),
        Dependency(key="searcher2", module="searcher", name="SDM"),
    ]

    commands = ["run"] + Task.help_commands
    default_command = "run"

    def run(self):
        output_dir = self.get_results_path()

        # read the title queries from the chosen benchmark's topic file
        results1 = self.searcher1.query_from_file(self.benchmark.topic_file, output_dir / "searcher1")
        results2 = self.searcher2.query_from_file(self.benchmark.topic_file, output_dir / "searcher2")
        searcher_results = [results1, results2]

        # using the benchmark's folds, which each contain train/validation/test queries,
        # choose the best run in `output_dir` for the fold based on the validation queries
        # and return metrics calculated on the test queries
        best_results = evaluator.search_best_run(
            searcher_results, self.benchmark, primary_metric=self.config["optimize"], metrics=evaluator.DEFAULT_METRICS
        )

        for fold, path in best_results["path"].items():
            shortpath = "..." + path[:-20]
            logger.info("fold=%s best run: %s", fold, shortpath)

        logger.info("cross-validated results when optimizing for '%s':", self.config["optimize"])
        for metric, score in sorted(best_results["score"].items()):
            logger.info("%15s: %0.4f", metric, score)

        return best_results

```

```eval_rst
.. note:: The module needs to be registered in order for Capreolus to find it. Registration happens when the ``@Task.register`` decorator is applied, so no additional steps are needed to use the new Task via the Python API. When using the Task via the CLI, the ``tutorial.py`` file containing it needs to be imported in order for the Task to be registered. This can be accomplished by placing the file inside the ``capreolus.tasks`` package (see ``capreolus.task.__path__``). However, in this case, the above Task is already provided with Capreolus as ``tasks/tutorial.py``.
```

Let's try running the Task we just declared via the Python API.

```python
>>> task = TutorialTask()
>>> results = task.run()
>>> results['score']['map']
0.1855654735508547
# looks like we got an improvement! which run was better?
>>> results['path']
...searcher_bm25(k1=0.9,b=0.4)_rm3(fbTerms=25,fbDocs=10,originalQueryWeight=0.5)'}
```

### Module APIs
Each module type's base class describes the module API that should be implemented to create new modules of that type.
Check out the API documentation to learn more:
<a href="autoapi/capreolus/benchmark/index.html">Benchmark</a>, 
<a href="autoapi/capreolus/collection/index.html">Collection</a>, 
<a href="autoapi/capreolus/extractor/index.html">Extractor</a>, 
<a href="autoapi/capreolus/index/index.html">Index</a>, 
<a href="autoapi/capreolus/reranker/index.html">Reranker</a>, 
<a href="autoapi/capreolus/searcher/index.html">Searcher</a>, 
<a href="autoapi/capreolus/task/index.html">Task</a>, 
<a href="autoapi/capreolus/tokenizer/index.html">Tokenizer</a>, and
<a href="autoapi/capreolus/trainer/index.html">Trainer</a>.


## Next Steps
- Learn more about [running pipelines using the command line interface](cli.md)
- View what [Capreolus modules](modules.md) are available
