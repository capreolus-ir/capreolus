# Running on TPUs

## Overview
All Tensorflow-compatible rerankers support training and inference on [Google TPUs](https://cloud.google.com/tpu). Capreolus has been tested with both v2-8 TPUs and v3-8 TPUs.

```eval_rst
.. warning:: TPUs stream their data from buckets in Google Cloud Storage rather than reading their input data from the local machine (i.e., the machine running Capreolus). Capreolus will automatically preprocess and upload the data to this bucket. However, note that GCS is not free and the user is responsible for manually deleting this data once it is no longer needed.
```

To use a TPU with a Tensorflow-compatible `Reranker` (i.e., a reranker that depends on the `tensorflow` <a href="autoapi/capreolus/trainer/index.html">Trainer</a> module), set the following config options:
- `tpuname`: the name of your TPU, such as *mytpu1*
- `tpuzone`: the cloud zone your TPU is in, such as *us-central1-f*
- `storage`: path to a GCS bucket where data should be stored, such as *gs://your-bucket/abc/*
- recommended: set `usecache=True` with the trainer and extractor

After setting these options, you can run Capreolus as normal. Watch for INFO logging messages at the beginning of training to confirm the TPU is being used.

```eval_rst
.. note:: While any Tensorflow-compatible `Reranker` can be used with TPUs, this will actually slow down small models like KNRM. TPUs are most useful with large Transformer-based models.
```

## Models
The following models are good candidates for running on TPUs:

```eval_rst
.. autoapiclass:: capreolus.reranker.TFBERTMaxP.TFBERTMaxP
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```

```eval_rst
.. autoapiclass:: capreolus.reranker.parade.TFParade
   :noindex:
   
   .. autoapiattribute:: module_name
      :noindex:
```
