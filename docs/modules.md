# Available Modules

The `Benchmark`, `Reranker`, and `Searcher` module types are most often configured by the end user.
For a complete list of modules, run the command `capreolus modules` or see the <a href="autoapi/capreolus/index.html">API Reference</a>.

```eval_rst
.. important:: When using Capreolus' configuration system, modules are selected by specifying their ``module_name``.
   For example, the ``NF`` benchmark can be selected with the ``benchmark.name=nf`` config string, the equivalent config dictionary ``{"benchmark": {"name": "nf"}}``, or imported as the class ``benchmark.nf.NF``.
```

## Benchmarks

### ANTIQUE
```eval_rst
.. autoapiclass:: capreolus.benchmark.antique.ANTIQUE
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```


### CodeSearchNet
```eval_rst
.. autoapiclass:: capreolus.benchmark.codesearchnet.CodeSearchNetCorpus
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

```eval_rst
.. autoapiclass:: capreolus.benchmark.codesearchnet.CodeSearchNetChallenge
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### (TREC) COVID

```eval_rst
.. autoapiclass:: capreolus.benchmark.covid.COVID
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```


### Dummy

```eval_rst
.. autoapiclass:: capreolus.benchmark.dummy.DummyBenchmark
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### NF Corpus

```eval_rst
.. autoapiclass:: capreolus.benchmark.nf.NF
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### (TREC) Robust04
```eval_rst
.. autoapiclass:: capreolus.benchmark.robust04.Robust04
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
      
   .. autoapiattribute:: dependencies
      :noindex:
```

```eval_rst
.. autoapiclass:: capreolus.benchmark.robust04.Robust04Yang19
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
      
   .. autoapiattribute:: dependencies
      :noindex:

```

## Searchers

### BM25
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.BM25
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### BM25 with RM3 expansion
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.BM25RM3
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

## Rerankers
Coming soon
