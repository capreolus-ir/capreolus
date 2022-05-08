# Capreolus: monoBERT Ranking Baselines on MS MARCO Passage Retrieval 

This page contains instructions for running monoELECTRA with LCE loss baselines on MS MARCO passage (v1) ranking task using Capreolus.
Basically reproduce the results in [this](to-be-added) paper.

For the set-up and monoBERT w/ hinge loss experiments, please refer to [this](MS_MARCO.md) page

## Running MS MARCO 
The config file (config_msmarco_lce.txt)[config_msmarco_lce.txt] could be used out-of-box, with the following command: 

```bash
python -m capreolus.run rerank.train with file=docs/reproduction/config_msmarco_lce.txt
```

The config would achieve `MRR@10` around `0.395~0.4` (maybe <0.01 points fluctuation).
It trains monoELECTRA with the hard negative data prepared from the [TCT-ColBERT](https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf), and uses LCE loss with 3 hard negative per query.
To experiments with different hard negative example, simply spcify `sampler.nneg`. 
For example, the following command would run the same config but with 7 hard negatives per query,
which should gives `MRR@10` around `0.405~0.41` 
```bash
python -m capreolus.run rerank.train with file=docs/reproduction/config_msmarco_lce.txt sampler.nneg=7
```

## Replication Logs
+ Results replicated by [@crystina-z](https://github.com/crystina-z) on 2022-01-13 (commit [`d377798`](https://github.com/crystina-z/capreolus-1/commit/6c3759fe620f18f8939670176a18c744752bc9240)) (2 Quadro RTX 8000, each 48G memory)