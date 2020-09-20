#!/bin/bash

source /GW/PKB/work/ghazaleh/anaconda3/bin/activate myenv
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/GW/PKB/work/ghazaleh/capreolus/ ;

domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
bm25c=$5
dataset=kitt
filterq=DONT

FOLDNUM=$(( ((SLURM_ARRAY_TASK_ID-1)%10)+1 ))
if ((SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 10)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    filterq=topic-alltopics_tf_k-1
  fi
fi
if ((SLURM_ARRAY_TASK_ID >= 11 && SLURM_ARRAY_TASK_ID <= 20)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    filterq=topic-amazon_tf_k-1
  fi
fi
if ((SLURM_ARRAY_TASK_ID >= 21 && SLURM_ARRAY_TASK_ID <= 30)); then
  if [ "$querytype" != "query" ]; then
    filterq=user-allusers_tf_k-1
  fi
fi
if ((SLURM_ARRAY_TASK_ID >= 31 && SLURM_ARRAY_TASK_ID <= 40)); then
  if [ "$querytype" != "query" ]; then
    filterq=user-amazon_tf_k-1
  fi
fi

echo "$domain - $pipeline - $querytype - $entitystrategy - $filterq - $FOLDNUM"

if [ "$filterq" != "DONT" ]; then
  if [ "$entitystrategy" == "noneE" ]; then
    if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
    fi
  fi
  if [ "$entitystrategy" == "allE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "domainE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "specE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "onlyNE" ]; then
   time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "domainOnlyNE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "specOnlyNE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=$bm25c collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.onlyNamedEntities=True reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
fi
conda deactivate