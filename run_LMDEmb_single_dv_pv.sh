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
entitystrategy=$3
domainvocsp=$4
dataset=kitt
filterq=DONT

declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

qtidx=$(( (SLURM_ARRAY_TASK_ID-1)/40 ))
querytype=${profiles[$qtidx]}

pvidx=$(( SLURM_ARRAY_TASK_ID - (qtidx * 40)  ))

FOLDNUM=$(( ((pvidx-1)%10)+1 ))
if ((pvidx >= 1 && pvidx <= 10)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    filterq=topic-alltopics_tf_k-1
  fi
fi
if ((pvidx >= 11 && pvidx <= 20)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    filterq=topic-amazon_tf_k-1
  fi
fi
if ((pvidx >= 21 && pvidx <= 30)); then
  if [ "$querytype" != "query" ]; then
    filterq=user-allusers_tf_k-1
  fi
fi
if ((pvidx >= 31 && pvidx <= 40)); then
  if [ "$querytype" != "query" ]; then
    filterq=user-amazon_tf_k-1
  fi
fi

echo "$domain - $pipeline - $querytype - $entitystrategy - $domainvocsp - $filterq - $FOLDNUM"

if [ "$filterq" != "DONT" ]; then
  if [ "$entitystrategy" == "noneE" ]; then
    if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
    fi
  fi
  if [ "$entitystrategy" == "allE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "domainE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "specE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "onlyNE" ]; then
   time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "domainOnlyNE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
  if [ "$entitystrategy" == "specOnlyNE" ]; then
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=specific_domainrel reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domainrelatedness.return_top=10 reranker.extractor.entityspecificity.return_top=5 reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.filter_query=$filterq fold=s$FOLDNUM ;
  fi
fi

conda deactivate
