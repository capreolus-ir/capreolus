#!/bin/bash
source /home/ghazaleh/anaconda3/bin/activate venvcuda
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/home/ghazaleh/capreolus_dev/capreolus/ ;

domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
assessed_set=$5
querycut=$6
dataset=kitt
FOLDNUM=$SLURM_ARRAY_TASK_ID
echo "$domain - $pipeline - $querytype - $entitystrategy - $assessed_set -$FOLDNUM"

if [ "$entitystrategy" == "noneE" ]; then
  if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
    time python -m capreolus.run rerank.train with searcher=qrels reranker=KNRM collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut fold=s$FOLDNUM ;
  fi
fi
conda deactivate

