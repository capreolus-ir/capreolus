# Capreolus: Reranking robust04 with PARADE
This page contains instructions for running Capreolus' PARADE implementation on the robust04 ad-hoc retrieval benchmark.

## Setup
This section contains instructions for installing Capreolus. **Do not** install Capreolus via pip, because we want a copy of the `master` branch that can be modified locally.

1. Ensure Python 3.6+ and Java 11 are installed. [See the installation guide for help.](https://capreolus.ai/en/latest/installation.html)
2. [Install PyTorch 1.6.0.](https://pytorch.org/get-started/locally/) If possible, choose CUDA 10.1 to match our environment.
3. Clone the Capreolus repository: `git clone https://github.com/capreolus-ir/capreolus`
4. You should now have a `capreolus` folder that contains various files as well as another `capreolus` folder, which contains the actual capreolus Python package. This is a [common layout for Python packages](https://python-packaging.readthedocs.io/en/latest/minimal.html); the inside folder (i.e., `capreolus/capreolus`) corresponds to the Python package.
5. Install dependencies: `pip install -r capreolus/requirements.txt`

## Testing installation
1. To run capreolus, `cd` into the top-level `capreolus` directory  and run `python -m capreolus.run`. This is equivalent to the `capreolus` command available when pip-installed. You should see a help message.
2. Let's try one more command to ensure everything is setup correctly: `python -m capreolus.run rank.print_config`. This should print a description of the default ranking config.
3. Briefly read about [configuring Capreolus](https://capreolus.ai/en/latest/installation.html#configuring-capreolus). The main thing to note is that results will be stored in `~/.capreolus` by default.

## Running PARADE (reduced memory usage)
This section describes how to run PARADE on a GPU with 16GB RAM. This is substantially less than used in the [paper](https://arxiv.org/abs/2008.09093), so we'll train on a single fold and change many hyperparameters to make this run smoothly. However, this won't reach the same effectiveness as the full PARADE model (see instructions below).

1. Make sure you have an available GPU and are in the top-level `capreolus` directory.
2. Train and evaluate PARADE on a single fold: `python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade_small.txt fold=s1`
3. This command takes about 3.5 hours on a Titan Xp GPU. Once it finishes, metrics on the dev and test sets are shown:
> 2020-10-20 12:39:37,265 - INFO - capreolus.task.rerank.evaluate - rerank: fold=s1 dev metrics: P_1=0.688 P_10=0.529 P_20=0.428 P_5=0.596 judged_10=0.998 judged_20=0.995 judged_200=0.947 map=0.271 ndcg_cut_10=0.545 ndcg_cut_20=0.504 ndcg_cut_5=0.577 recall_100=0.453 recall_1000=0.453 recip_rank=0.787

> 2020-10-20 12:39:37,343 - INFO - capreolus.task.rerank.evaluate - rerank: fold=s1 test metrics: P_1=0.532 P_10=0.472 P_20=0.418 P_5=0.528 judged_10=0.989 judged_20=0.989 judged_200=0.931 map=0.285 ndcg_cut_10=0.470 ndcg_cut_20=0.471 ndcg_cut_5=0.485 recall_100=0.490 recall_1000=0.490 recip_rank=0.672
4. Compare your *fold=s1* results to those shown here. Do they match? If so, we can move on to reproducing the full PARADE model.

## Running PARADE (full model with normal memory usage)
This requires a 48GB GPU or a TPU. It has been tested on NVIDIA Quadro RTX 8000s and Google Cloud TPUs.

1. Make sure you have an available GPU and are in the top-level `capreolus` directory.
2. Train and evaluate PARADE on each of the five robust04 folds (splits *s1-s5*): <br/>
  This can be done with TensorFlow: <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade.txt fold=s1` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade.txt fold=s2` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade.txt fold=s3` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade.txt fold=s4` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade.txt fold=s5` <br/>
  Or PyTorch: <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_paradept.txt fold=s1` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_paradept.txt fold=s2` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_paradept.txt fold=s3` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_paradept.txt fold=s4` <br/>
`python -m capreolus.run rerank.traineval with file=docs/reproduction/config_paradept.txt fold=s5`
3. Each command will take a long time; approximately 36 hours on a Quadro 8000 (much faster on TPU). As above, per-fold metrics are displayed after each fold completes.
4. When the final fold completes, cross-validated metrics are also displayed.

Heads-up: while the above PyTorch commands works for PyTorch versions from `1.6` to `1.8`, we observed the score is a bit lower with `1.8`:
`torch` version | mAP | P@20 | NDCG@20
-- | -- | -- | --
1.6 | 0.3687 | 0.4851 | 0.5533
1.7 | 0.3687 | 0.4851 | 0.5533
1.8 | 0.3666 | 0.4783 | 0.5478
