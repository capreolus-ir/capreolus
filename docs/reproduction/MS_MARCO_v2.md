# Capreolus: MaxP Reranking Baselines on MS MARCO v2 Retrieval

This page contains instructions for running MaxP baselines on MS MARCO v2 document and passage ranking task using Capreolus.
After [setting up](./PARADE.md#Setup) capreolus and [testing the installation](./PARADE.md#testing-installation), we are ready to start the replication.
Note that all the following scripts requires that we are on the top-level `capreolus` folder.


## Document Retrieval
1. Dataset preparation
```
mkdir -p capreolus/data/msdoc_v2
# data downloading
cat docv2_dev_qrels.tsv  docv2_train_qrels.tsv > qrels.txt
cat docv2_dev_queries.tsv  docv2_train_queries.tsv > topics.txt
```
Now the `capreolus/data/msdoc_v2` should contain the following files:
```
- docv2_dev_qrels.tsv
- docv2_dev_queries.tsv
- docv2_train_qrels.tsv
- docv2_train_queries.tsv
- docv2_train_top100.txt
- qrels.txt
- topics.txt
- msmarco_v2_doc/ # which contains 60 .gz files
```
Note: if on `ceres`, the data is available under `/GW/carpet/nobackup/czhang/msdoc_v2`:
```
ln -s /GW/carpet/nobackup/czhang/msdoc_v2 capreolus/data
```

2. To run BERT-MaxP with 10 passage:
```
collection_name=msdoc_v2
n_passages=10

python -m capreolus.run rerank.train with \
    file=docs/reproduction/config_msmarco_v2.txt \
    reranker.trainer.amp=True \
    reranker.extractor.numpassages=$n_passages \
    benchmark.collection.name=$collection_name
```
Expected score:
```
MAP     MRR      R@100
0.2646  0.2677   0.5956
```


## Passage Retrieval
1. Dataset preparation
```
mkdir -p capreolus/data/mspass_v2
# data downloading
cat passv2_dev_qrels.tsv  passv2_train_qrels.tsv > qrels.txt
cat passv2_dev_queries.tsv  passv2_train_queries.tsv > topics.txt
```
Now the `capreolus/data/mspass_v2` should contain the following files:
```
- passv2_dev_qrels.tsv
- passv2_dev_queries.tsv
- passv2_train_qrels.tsv
- passv2_train_queries.tsv
- passv2_train_top100.txt
- qrels.txt
- topics.txt
- msmarco_v2_passage/ # which contains 70 .gz files
```
Note: if on `ceres`, the data is available under `/GW/carpet/nobackup/czhang/mspass_v2`:
```
ln -s /GW/carpet/nobackup/czhang/mspass_v2 capreolus/data
```

2. To run monoBERT:
```
collection_name=mspsg_v2
n_passages=1

python -m capreolus.run rerank.train with \
    file=docs/reproduction/config_msmarco_v2.txt \
    reranker.trainer.amp=True \
    reranker.extractor.numpassages=$n_passages \
    benchmark.collection.name=$collection_name
```

Expected score:
```
MAP      MRR     R@100
0.1503   0.152   0.3397
```