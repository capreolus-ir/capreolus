# Capreolus: monoBERT Ranking Baselines on MS MARCO Passage Retrieval 

This page contains instructions for running MaxP baselines on MS MARCO passage ranking task using Capreolus.
If you are a Compute Canada user, 
first follow [this](../setup/setup-cc.md) guide to set up the environment on CC then continue with this page.

Once the environment is set, you can verify the installation with [these instructions](./PARADE.md#testing-installation).

## Explore LR Scheduler Setting
Below are the possible lr scheduler values (combination) you can pick up and run: <br/> 

| No.         |   | bertlr | lr   | itersize | warmupsteps | decaystep | decaytype |
|-------------|---|--------|------|----------|-------------|-----------|-----------|
| 0 (default) |   | 1e-5   | 1e-3 |    40000 |           0 |         0 | None      |
| 1           |   | 1e-5   | 1e-5 |    40000 |           0 |         0 | None      |
| 2           |   | 1e-6   | 1e-6 |    40000 |           0 |         0 | None      |
| 3           |   | 1e-5   | 1e-3 |    40000 |        4000 |         0 | None      |
| 4           |   | 1e-5   | 1e-5 |    40000 |        4000 |         0 | None      |
| 5           |   | 1e-6   | 1e-6 |    40000 |        4000 |         0 | None      |
| 6           |   | 1e-5   | 1e-3 |    40000 |           0 |     40000 | linear    |
| 7           |   | 1e-5   | 1e-5 |    40000 |           0 |     40000 | linear    |
| 8           |   | 1e-6   | 1e-6 |    40000 |           0 |     40000 | linear    |
| 9           |   | 1e-5   | 1e-3 |    40000 |        4000 |     40000 | linear    |
| 10          |   | 1e-5   | 1e-5 |    40000 |        4000 |     40000 | linear    |
| 11          |   | 1e-6   | 1e-6 |    40000 |        4000 |     40000 | linear    |

## Running MS MARCO 
This requires GPU(s) with 48GB memory (e.g. 4 V100 or a RTX 8000) or a TPU. 
1. Make sure you are in the top-level `capreolus` directory; 
2. Train on MS MARCO Passage using the following scripts, 
    while replacing the lr scheduler variables with the one you picked up <br/> 
    ```
    # say you chose to run "decay"=0.1
    decay=0.1
    python -m capreolus.run rerank.train with \
        file=docs/reproduction/config_msmarco.txt  \
        reranker.trainer.decay=$decay \
        fold=s1
    ```
3. The command will take over 2 days on *4 V100s* for BERT-base, and even longer on *8 V100s* for BERT-large. 
    Per-fold metrics on dev set are displayed after completion, where `MRR@10` is the one to use for this task.
    (for CC users, BERT-large can only be run on `graham` `cedar`, as `beluga` does not have 8-GPU node.)

## Expected Scores
With the configurations in `docs/reproduction/config_msmarco.txt` and all the others as default, 
the MRR@10 on dev set should be around `0.339`. 

## Replication Logs
+ Results (with hypperparameter-0) replicated by [@crystina-z](https://github.com/crystina-z) on 2020-12-06 (commit [`6c3759f`](https://github.com/crystina-z/capreolus-1/commit/6c3759fe620f18f8939670176a18c744752bc9240)) (Tesla V100 on Compute Canada)
