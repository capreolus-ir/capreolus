#!/bin/bash
source /GW/PKB/work/ghazaleh/anaconda3/bin/activate myenv
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/home/ghazaleh/capreolus_dev/capreolus/ ;

domain=$1
pipeline=$2
entitystrategy=$3
assessed_set=$4
dataset=kitt
querycut=DONT

declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'chatprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

qtidx=$(( (SLURM_ARRAY_TASK_ID-1)/60 ))
querytype=${profiles[$qtidx]}

pvidx=$(( SLURM_ARRAY_TASK_ID - (qtidx * 60)  ))

FOLDNUM=$(( ((pvidx-1)%10)+1 ))
if ((pvidx >= 1 && pvidx <= 10)); then
  querycut=None
fi
if ((pvidx >= 11 && pvidx <= 20)); then
  querycut=unique_most_frequent
fi
if ((pvidx >= 21 && pvidx <= 30)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    querycut=unique_topic-alltopics
  fi
fi
if ((pvidx >= 31 && pvidx <= 40)); then
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    querycut=unique_topic-amazon
  fi
fi
if ((pvidx >= 41 && pvidx <= 50)); then
  if [ "$querytype" != "query" ]; then
    querycut=unique_user-allusers
  fi
fi
if ((pvidx >= 51 && pvidx <= 60)); then
  if [ "$querytype" != "query" ]; then
    querycut=unique_user-amazon
  fi
fi

echo "$domain - $pipeline - $querytype - $entitystrategy - $querycut - $assessed_set -$FOLDNUM"

if [ "$querycut" != "DONT" ]; then
  if [ "$entitystrategy" == "noneE" ]; then
    if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
      time python -m capreolus.run rerank.train with searcher=qrels reranker=KNRM collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut fold=s$FOLDNUM ;
    fi
  fi
fi
conda deactivate


