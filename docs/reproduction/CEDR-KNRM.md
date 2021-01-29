# Capreolus: Reranking robust04 with CEDR-KNRM
This page contains instructions for running CEDR-KNRM on the robust04 benchmark.

[*CEDR: Contextualized Embeddings for Document Ranking*](https://arxiv.org/pdf/1904.07094.pdf).
Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. SIGIR 2019.

## Setup
Install Capreolus v0.2.6 or later. See the [installation guide](https://capreolus.ai/en/latest/installation.html) for help installing a release. To install from GitHub, see the [PARADE guide](https://github.com/capreolus-ir/capreolus/blob/master/docs/reproduction/PARADE.md).

## Running CEDR-KNRM

This section describes how to run CEDR-KNRM on a GPU or TPU (Tensorflow only). This should work on any GPU that supports mixed precision (float16) and has at least 16 GB of RAM. 

When using a less powerful GPU or disabling mixed precision (`reranker.trainer.amp`), reduce the batch size (`reranker.trainer.batch` and `reranker.trainer.evalbatch`). When using a TPU or multiple GPUs (Tensorflow only), increase the batch and prediction batch sizes; a TPU v3-8 should allow `reranker.trainer.batch=64` and `reranker.trainer.evalbatch=1024`. See the [TPU instructions](https://capreolus.ai/en/latest/tpu.html) for other config options to set.

### PyTorch

1. Download `config_cedrknrm_pt.txt` [from GitHub](https://raw.githubusercontent.com/capreolus-ir/capreolus/master/docs/reproduction/config_cedrknrm_pt.txt) or identify the path to it in your local Capreolus installation.
2. Train and evaluate CEDR-KNRM on each of the five robust04 folds (splits *s1-s5*):<br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_pt.txt fold=s1` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_pt.txt fold=s2` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_pt.txt fold=s3` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_pt.txt fold=s4` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_pt.txt fold=s5`
3. Each command will take a few hours on a single V100 GPU. Per-fold metrics are displayed after each fold completes.
4. When the final fold completes, cross-validated metrics are also displayed.
 

### Tensorflow

1. Download `config_cedrknrm_tf.txt` [from GitHub](https://raw.githubusercontent.com/capreolus-ir/capreolus/master/docs/reproduction/config_cedrknrm_tf.txt) or identify the path to it in your local Capreolus installation.
2. Train and evaluate CEDR-KNRM on each of the five robust04 folds (splits *s1-s5*):<br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_tf.txt fold=s1` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_tf.txt fold=s2` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_tf.txt fold=s3` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_tf.txt fold=s4` <br/>
`python -m capreolus.run rerank.traineval with file=config_cedrknrm_tf.txt fold=s5`
3. Each command will take a few hours on a single V100 GPU. Per-fold metrics are displayed after each fold completes.
4. When the final fold completes, cross-validated metrics are also displayed.

Note that the Tensorflow implementation has primarily been tested on TPUs.

## Running BERT-KNRM, VanillaBERT, and other model variants

The CEDR-KNRM model can be converted to BERT-KNRM by omitting the CLS token or to "VanillaBERT" by omitting the KNRM component. To do so, add the following config options to the above commands:
- BERT-KNRM:   `reranker.simmat_layers=0..12,1 reranker.cls=None`
- VanillaBERT: `reranker.simmat_layers=-1 reranker.cls=avg`

Set the `reranker.pretrained` config option to choose between BERT and ELECTRA base models with or without pre-fine-tuning on MS MARCO. Possible values: `electra-base`, `electra-base-msmarco`, `bert-base-uncased`, and `bert-base-msmarco`.

Set the `reranker.extractor.numpassages` config option to change the number of passages considered. Increasing this past 4 will likely require decreasing the batch sizes.

See the config file for other relevant values.
