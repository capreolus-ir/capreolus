# Capreolus: MaxP Reranking Baselines on MS MARCO v2 Retrieval

## Dataset Preparation


## Document Retrieval
```
collection_name=msdoc_v2
n_passage=10

python -m capreolus.run rerank.train with \
    file=docs/reproduction/config_msmarco_v2.txt \
    reranker.trainer.amp=True \
    reranker.extractor.numpassages=n_passage \
    benchmark.collection.name=collection_name
```


## Passage Retrieval
```
collection_name=mspsg_v2
n_passage=1

python -m capreolus.run rerank.train with \
    file=docs/reproduction/config_msmarco_v2.txt \
    reranker.trainer.amp=True \
    reranker.extractor.numpassages=n_passage \
    benchmark.collection.name=collection_name
```