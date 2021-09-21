# Capreolus: monoBERT Ranking Baselines on MS MARCO Passage Retrieval 

This page contains instructions for running MaxP baselines on MS MARCO passage ranking task using Capreolus.
If you are a Compute Canada user, 
first follow [this](../setup/setup-cc.md) guide to set up the environment on CC then continue with this page.

Once the environment is set, you can verify the installation with [these instructions](./PARADE.md#testing-installation).

## Running MS MARCO 
This requires GPU(s) with 48GB memory (e.g. 3 V100 or a RTX 8000) or a TPU. 
1. Make sure you are in the top-level `capreolus` directory;
2. Use the following script to run a "mini" version of the MS MARCO fine-tuning, testing if everything is working. 
    ```bash
    python -m capreolus.run rerank.train with file=docs/reproduction/config_msmarco.txt
    ``` 
    This would train the monoBERT for only 3k steps with batch size to be 4, then rerank the *top100* documents per query. 
    The script should take no more than 24 hours to finish, and could be fit into a single `v100l`.
    At the end of execusion, it would display a bunch of metrics, where `MRR@10` should be around `0.295`.

3. Once the above is done, we can fine-tune a full version on MS MARCO Passage using the following scripts: 
    ```bash
    niters=10
    batch_size=16
    validatefreq=$niters # to ensure the validation is run only at the end of training
    decayiters=$niters   # either same with $itersize or 0
    threshold=1000       # the top-k documents to rerank

    python -m capreolus.run rerank.train with \
        file=docs/reproduction/config_msmarco.txt  \
        threshold=$threshold \
        reranker.trainer.niters=$niters \
        reranker.trainer.batch=$batch_size \
        reranker.trainer.decayiters=$decayiters \
        reranker.trainer.validatefreq=$validatefreq \
        fold=s1
    ```
    The data preparation time may vary a lot on different machines.
    After data is prepared, it would take 4~6 hours to train and 6～10 hours to inference with *4 V100s* for BERT-base. 
    This should achieve `MRR@10=0.35+`.

### For CC slurm users:
In case you are new to [slurm](https://slurm.schedmd.com/documentation.html), a sample slurm script for the *full version* fine-tuning could be found under `docs/reproduction/sample_slurm_script.sh`.
This should work on `cedar` directly via `sbatch sample_slurm_script.sh`.
To adapt it to the `mini` version, simply change the GPU number and request time into:
```
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=24:00:00
``` 

## Replication Logs
+ Results (with hypperparameter-0) replicated by [@crystina-z](https://github.com/crystina-z) on 2020-12-06 (commit [`6c3759f`](https://github.com/crystina-z/capreolus-1/commit/6c3759fe620f18f8939670176a18c744752bc9240)) (Tesla V100 on Compute Canada)
+ Results (with hypperparameter-6) replicated by [@Dahlia-Chehata](https://github.com/Dahlia-Chehata) on 2021-03-29 (commit [`7915aad`](https://github.com/capreolus-ir/capreolus/commit/7915aad75406527a3b88498926cff85259808696)) (Tesla V100 on Compute Canada)
+ Results (with hypperparameter-7) replicated by [@larryli1999](https://github.com/larryli1999) on 2021-05-16 (commit [`6d1aed2`](https://github.com/capreolus-ir/capreolus/commit/6d1aed29de7828ceb94560a8bf7c87f1af5458b5)) (Tesla V100 on Compute Canada)
+ Results (MRR@10=0.356) replicated by [@andrewyguo](https://github.com/andrewyguo) on 2021-05-29 (commit [`1ce71d9`](https://github.com/capreolus-ir/capreolus/commit/1ce71d93ab5473b40d4ae02768fd053261b27320)) (Tesla V100 on Compute Canada)
+ Results (MRR@10=0.345) replicated by [@leungjch](https://github.com/leungjch) on 2021-09-21 (commit [`3521171`](https://github.com/capreolus-ir/capreolus/commit/3521171ecf38cebfec5e19e22621bf9dfabf58d9)) (4x Tesla V100, 16 CPUs, 120GB RAM on Compute Canada)
