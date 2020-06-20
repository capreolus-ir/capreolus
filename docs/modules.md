# Available Modules

The `Benchmark`, `Reranker`, and `Searcher` module types are most often configured by the end user.
For a complete list of modules, run the command `capreolus modules` or see the <a href="autoapi/capreolus/index.html">API Reference</a>.

```eval_rst
.. important:: When using Capreolus' configuration system, modules are selected by specifying their ``module_name``.
   For example, the ``NF`` benchmark can be selected with the ``benchmark.name=nf`` config string or the equivalent config dictionary ``{"benchmark": {"name": "nf"}}``.
   
   The corresponding class can be created as ``benchmark.nf.NF(config=..., provide=...)`` or created by name with ``Benchmark.create("nf", config=..., provide=...)``.
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
```

```eval_rst
.. autoapiclass:: capreolus.benchmark.robust04.Robust04Yang19
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

## Searchers

```eval_rst
.. note:: Some searchers (e.g., BM25) automatically perform a cross-validated grid search when their parameters are provided as lists. For example, ``searcher.b=0.4,0.6,0.8 searcher.k1=1.0,1.5``.
```

### BM25
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.BM25
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### BM25 with Axiomatic expansion
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.AxiomaticSemanticMatching
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


### BM25 PRF
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.BM25PRF
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```


### F2Exp
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.F2Exp
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### F2Log
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.F2Log
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### I(n)L2
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.INL2
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### QL with Dirichlet smoothing
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.DirichletQL
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### QL with J-M smoothing
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.QLJM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### SDM
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.SDM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### SPL
```eval_rst
.. autoapiclass:: capreolus.searcher.anserini.SPL
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

## Rerankers

```eval_rst
.. note:: Rerankers are implemented in PyTorch or TensorFlow. Rerankers with TensorFlow implementations can run on both GPUs and TPUs.
```


### CDSSM
```eval_rst
.. autoapiclass:: capreolus.reranker.CDSSM.CDSSM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### ConvKNRM
```eval_rst
.. autoapiclass:: capreolus.reranker.ConvKNRM.ConvKNRM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### DRMM
```eval_rst
.. autoapiclass:: capreolus.reranker.DRMM.DRMM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### DRMMTKS
```eval_rst
.. autoapiclass:: capreolus.reranker.DRMMTKS.DRMMTKS
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### DSSM
```eval_rst
.. autoapiclass:: capreolus.reranker.DSSM.DSSM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### DUET
```eval_rst
.. autoapiclass:: capreolus.reranker.DUET.DUET
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### DeepTileBars
```eval_rst
.. autoapiclass:: capreolus.reranker.DeepTileBar.DeepTileBar
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### HiNT
```eval_rst
.. autoapiclass:: capreolus.reranker.HINT.HINT
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### KNRM
```eval_rst
.. autoapiclass:: capreolus.reranker.KNRM.KNRM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### PACRR
```eval_rst
.. autoapiclass:: capreolus.reranker.PACRR.PACRR
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### POSITDRMM
```eval_rst
.. autoapiclass:: capreolus.reranker.POSITDRMM.POSITDRMM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### TK
```eval_rst
.. autoapiclass:: capreolus.reranker.TK.TK
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### TensorFlow KNRM
```eval_rst
.. autoapiclass:: capreolus.reranker.TFKNRM.TFKNRM
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

### TensorFlow VanillaBERT
```eval_rst
.. autoapiclass:: capreolus.reranker.TFVanillaBert.TFVanillaBERT
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

