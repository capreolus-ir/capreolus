#!/bin/bash

source /GW/home-12/ghazaleh/anaconda3/bin/activate venc
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/NeuralIR/nobackup/ghazaleh/results_11092020/ ;
export CAPREOLUS_CACHE=/GW/NeuralIR/nobackup/ghazaleh/cache_11092020/ ;
export PYTHONPATH=/home/ghazaleh/Projects_Workspace_new/capreolus/ ;

domain=$1
pipeline=$2
querytype=$3
FOLDNUM=$4
entitystrategy=$5
domainvocsp=$6
dataset=kitt

if [ "$entitystrategy" == "noneE" ]; then
  if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
  fi
fi
if [ "$entitystrategy" == "allE" ]; then
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi
if [ "$entitystrategy" == "domainE" ]; then
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi
if [ "$entitystrategy" == "specE" ]; then
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi
if [ "$entitystrategy" == "onlyNE" ]; then
 time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi
if [ "$entitystrategy" == "domainOnlyNE" ]; then
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi
if [ "$entitystrategy" == "specOnlyNE" ]; then
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
fi