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
This section describes how to run PARADE on a GPU with 16GB RAM. This is substantially less than used in the [paper](https://arxiv.org/abs/2008.09093), so we'll reduce the batch size and the size of each passage to make the data to fit.

1. Make sure you have an available GPU and are in the top-level `capreolus` directory.
2. Train and evaluate PARADE on a single fold: `python -m capreolus.run rerank.traineval with file=docs/reproduction/config_parade_small.txt fold=s1`
3. This command takes about 3.5 hours on a Titan Xp GPU. Once it finishes, metrics on the dev and test sets are shown:
> 2020-09-01 15:45:10,053 - INFO - capreolus.task.rerank.evaluate - rerank: fold=s1 dev metrics: P_1=0.750 P_10=0.500 P_20=0.443 P_5=0.554 judged_10=0.992 judged_20=0.989 judged_200=0.947 map=0.267 ndcg_cut_10=0.533 ndcg_cut_20=0.513 ndcg_cut_5=0.562 recall_100=0.453 recall_1000=0.453 recip_rank=0.817

> 2020-09-01 15:45:10,095 - INFO - capreolus.task.rerank.evaluate - rerank: fold=s1 test metrics: P_1=0.596 P_10=0.487 P_20=0.419 P_5=0.549 judged_10=0.989 judged_20=0.985 judged_200=0.931 map=0.285 ndcg_cut_10=0.491 ndcg_cut_20=0.486 ndcg_cut_5=0.518 recall_100=0.490 recall_1000=0.490 recip_rank=0.727
4. Compare your *fold=s1* results to those shown here. Do they match? If so, we can move on to reproducing the full PARADE model.

## Running PARADE (full model with normal memory usage)
TODO. This requires a 48GB GPU, a TPU, or porting PARADE to Pytorch so we can iterate over passages rather than loading all of them in memory at once (see issue #86). The corresponding config is in `docs/reproduction/config_parade.txt`.
