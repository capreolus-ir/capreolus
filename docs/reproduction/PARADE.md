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

## Running PARADE
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

## Expected results
Note that results will vary slightly with your environment. 

 Environment | mAP | P@20 | NDCG@20
-- | -- | -- | --
Pytorch 1.6 (GPU) | 0.3687 | 0.4851 | 0.5533
Pytorch 1.7 (GPU) | 0.3687 | 0.4851 | 0.5533
Pytorch 1.8 (GPU) | 0.3666 | 0.4783 | 0.5478
Tensorflow 2.4 (TPU) | 0.3722 | 0.4783 | 0.5528
Tensorflow 2.5 (TPU) | 0.3626 | 0.4739 | 0.5449
