#!/bin/bash
source /GW/PKB/work/ghazaleh/anaconda3/bin/activate myenv
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_test/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_test/ ;
export PYTHONPATH=/home/ghazaleh/capreolus_dev/capreolus/ ;

domain=$1
pipeline=$2
querytype=$3
dataset=kitt
FOLDNUM=$SLURM_ARRAY_TASK_ID
echo "$domain - $pipeline - $querytype - $entitystrategy - $FOLDNUM"

time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=KNRM collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM ;
