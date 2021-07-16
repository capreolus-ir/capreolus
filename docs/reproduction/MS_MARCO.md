# Capreolus: monoBERT Ranking Baselines on MS MARCO Passage Retrieval 

This page contains instructions for running MaxP baselines on MS MARCO passage ranking task using Capreolus.
If you are a Compute Canada user, 
first follow [this](../setup/setup-cc.md) guide to set up the environment on CC then continue with this page.

Once the environment is set, you can verify the installation with [these instructions](./PARADE.md#testing-installation).

## Explore LR Scheduler Setting
Below are the possible lr scheduler values (combination) you can pick up and run: <br/> 

| No.         |   | bertlr | lr   | itersize | warmupsteps | decaystep | decaytype | Expected |
|-------------|---|--------|------|----------|-------------|-----------|-----------|----------|
| 0 (default) |   | 2e-5   | 1e-3 |    30000 |           0 |         0 | None      | 0.33+    | 
| 1           |   | 2e-5   | 2e-5 |    30000 |           0 |         0 | None      ||
| 2           |   | 2e-5   | 1e-3 |    30000 |        3000 |         0 | None      ||
| 3           |   | 2e-5   | 2e-5 |    30000 |        3000 |         0 | None      ||
| 4           |   | 2e-5   | 1e-3 |    30000 |           0 |     30000 | linear    ||
| 5           |   | 2e-5   | 2e-5 |    30000 |           0 |     30000 | linear    ||
| 6           |   | 2e-5   | 1e-3 |    30000 |        3000 |     30000 | linear    | 0.35+    |
| 7           |   | 2e-5   | 2e-5 |    30000 |        3000 |     30000 | linear    | 0.35+    |
| 8           |   | 3e-5   | 1e-3 |    30000 |        3000 |     30000 | linear    ||
| 9           |   | 3e-5   | 3e-5 |    30000 |        3000 |     30000 | linear    ||

## Running MS MARCO 
This requires GPU(s) with 48GB memory (e.g. 3 V100 or a RTX 8000) or a TPU. 
1. Make sure you are in the top-level `capreolus` directory; 
2. Train on MS MARCO Passage using the following scripts, 
    while replacing the lr scheduler variables with the one you picked up <br/> 
    ```
    lr=1e-3
    bertlr=2e-5   
    itersize=30000
    warmupsteps=3000
    decaystep=$itersize  # either same with $itersize or 0
    decaytype=linear
   
    python -m capreolus.run rerank.train with \
        file=docs/reproduction/config_msmarco.txt  \
        reranker.trainer.lr=$lr \
        reranker.trainer.bertlr=$bertlr \
        reranker.trainer.itersize=$itersize \
        reranker.trainer.warmupsteps=$warmupsteps \
        reranker.trainer.decaystep=$decaystep \
        reranker.trainer.decaytype="linear" \
        fold=s1
    ```
3.  Without data preparation, it will take 4~6 hours to train and 8～10 hours to inference on *4 V100s* for BERT-base, 
    and longer on for BERT-large. 
    Per-fold metrics on dev set are displayed after completion, where `MRR@10` is the one to use for this task.
    (for CC users, BERT-large can only be run with batch size 16 on `graham` `cedar`, 
    as each node on `beluga` has 16GB memory at maximum) 

## Replication Logs
+ Results (with hypperparameter-0) replicated by [@crystina-z](https://github.com/crystina-z) on 2020-12-06 (commit [`6c3759f`](https://github.com/crystina-z/capreolus-1/commit/6c3759fe620f18f8939670176a18c744752bc9240)) (Tesla V100 on Compute Canada)
+ Results (with hypperparameter-6) replicated by [@Dahlia-Chehata](https://github.com/Dahlia-Chehata) on 2021-03-29 (commit [`7915aad`](https://github.com/capreolus-ir/capreolus/commit/7915aad75406527a3b88498926cff85259808696)) (Tesla V100 on Compute Canada)
+ Results (with hypperparameter-7) replicated by [@larryli1999](https://github.com/larryli1999) on 2021-05-16 (commit [`6d1aed2`](https://github.com/capreolus-ir/capreolus/commit/6d1aed29de7828ceb94560a8bf7c87f1af5458b5)) (Tesla V100 on Compute Canada)
+ Results (MRR@10=0.356) replicated by [@andrewyguo](https://github.com/andrewyguo) on 2021-05-29 (commit [`1ce71d9`](https://github.com/capreolus-ir/capreolus/commit/1ce71d93ab5473b40d4ae02768fd053261b27320)) (Tesla V100 on Compute Canada)