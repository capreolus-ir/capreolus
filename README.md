[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Worfklow](https://github.com/capreolus-ir/capreolus/workflows/pytest/badge.svg)](https://github.com/capreolus-ir/capreolus/actions)
[![Documentation Status](https://readthedocs.org/projects/capreolus/badge/?version=latest)](https://capreolus.readthedocs.io/?badge=latest)
[![PyPI version fury.io](https://badge.fury.io/py/capreolus.svg)](https://pypi.python.org/pypi/capreolus/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 

# Capreolus
[![Capreolus](https://people.mpi-inf.mpg.de/~ayates/capreolus/capreolus-100px.png)](https://capreolus.ai) <br/>
Capreolus is a toolkit for conducting end-to-end ad hoc retrieval experiments. Capreolus provides fine control over the entire experimental pipeline through the use of interchangeable and configurable modules.

[Get started with a Notebook](https://colab.research.google.com/drive/161FnmLt3PgIXG-Z5eNg45z2iSZucVAnr?usp=sharing)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/161FnmLt3PgIXG-Z5eNg45z2iSZucVAnr?usp=sharing)

[Read the documentation for a detailed overview.](http://capreolus.ai/)

## Quick Start
1. Prerequisites: Python 3.7+ and Java 11. See the [installation instructions](https://capreolus.ai/en/latest/installation.html)
2. Install the pip package: `pip install capreolus`
3. Train a model: `capreolus rerank.traineval with benchmark.name=nf reranker.name=KNRM reranker.trainer.niters=2`
4. If the `train` command completed successfully, you've trained your first Capreolus reranker on the [NFCorpus dataset](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)! This command created several outputs, such as model checkpoints and TREC-format run files. To learn about these files, [read about running experiments with Capreolus](http://capreolus.ai/en/latest/cli.html).
5. To learn about different configuration options, try: `capreolus rerank.print_config with benchmark.name=nf reranker.name=KNRM`
6. To learn about different modules you can use, such as `reranker.name=DRMM`, try: `capreolus modules`
7. Learn about [running experiments via the Python API](https://capreolus.ai/en/latest/quick.html)

## Environment Variables
Capreolus uses environment variables to indicate where outputs should be stored and where document inputs can be found. Consult the table below to determine which variables should be set. Set them either on the fly before running Capreolus (`export CAPREOLUS_RESULTS=...`) or by editing your shell's initialization files (e.g., `~/.bashrc` or `~/.zshrc`).

| Environment Variable          | Default Value | Purpose |
|-------------------------------|---------------|---------|
| `CAPREOLUS_RESULTS`             | ~/.capreolus/results/    | Directory where results will be stored   |
| `CAPREOLUS_CACHE`               | ~/.capreolus/cache/      | Directory used for cache files |
| `CUDA_VISIBLE_DEVICES`          | (unset)     | Indicates GPUs available to PyTorch, starting from 0. For example, set to '1' the system's 2nd GPU (as numbered by `nvidia-smi`). Set to '' (an empty string) to force CPU.


