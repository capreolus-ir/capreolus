# Python API

Capreolus exposes an API that supports functionality similar to its command line interface (CLI).
This API is currently a work-in-progress that will be expanded in the next release.
This page assumes the reader is already familiar with [running experiments using Capreolus' CLI](cli.md).

## Basic usage
```python
import capreolus
pipeline = capreolus.train_pipeline({"reranker": "KNRM", "niters": 2, "benchmark": "robust04.title"})
```

The `train_pipeline` method trains a reranking pipeline for the specified number of iterations (`niters`). This function expects a config dict as an argument, with keys and values corresponding exactly to those provided on the command line. 

The `train_pipeline` method returns a `Pipeline` object describing the pipeline that was run. In the above example, `pipeline.reranker` would correspond to a trained `reranker.KNRM` object and `pipeline.reranker_path` indicates the path where output was stored. The pipeline's outputs are the same as when using the [CLI's train command](cli.html#train-command).


## Pipeline config
The configuration dict accepted by `train_pipeline` accepts the same config options as the CLI.
As with the CLI, any missing config options are filled in with reasonable defaults.

Train a pipeline:
```python
config = {
  "reranker": "KNRM",
  "benchmark": "robust04.title.wsdm20demo",
  "niters": 10,
  "expid": "testing",
}
pipeline = capreolus.train_pipeline(config)
```

Retrieve the full config used, after missing keys have been filled in with defaults:
```python
>>> print(pipeline.cfg)
{collection': 'robust04', ... 'reranker': KNRM', ... 'embeddings': 'glove6b', ... }
```


## Evaluating a trained model
```python
pipeline = capreolus.train_pipeline(config)

# results will be a dict of the form: {'map': 0.3423, 'ndcg': '0.12312', ...}
results = capreolus.evaluate_pipeline(pipeline)
```
