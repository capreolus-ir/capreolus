#!/bin/bash
source /GW/PKB/work/ghazaleh/anaconda3/bin/activate myenv
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_30092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_30092020/ ;
export PYTHONPATH=/GW/PKB/work/ghazaleh/capreolus/ ;

declare -a profiles=('basicprofileMR' 'chatprofileMR' 'query' 'basicprofile_general' 'chatprofile_general')

domain=$1
pipeline=$2
entitystrategy=$3
doccut=$4
assessed_set=$5
step=$6
dataset=kitt
querycut=DONT

qtidx=$(( (SLURM_ARRAY_TASK_ID-1)/20 ))
querytype=${profiles[$qtidx]}

pvidx=$(( SLURM_ARRAY_TASK_ID - (qtidx * 20)  ))

FOLDNUM=$(( ((pvidx-1)%10)+1 ))
if ((pvidx >= 1 && pvidx <= 10)); then
  querycut=None
fi
if ((pvidx >= 11 && pvidx <= 20)); then
  querycut=unique_most_frequent
fi

echo "$domain - $pipeline - $querytype - $entitystrategy - $querycut - $doccut - $assessed_set -$FOLDNUM"

if [ "$querycut" != "DONT" ]; then
  if [ "$entitystrategy" == "noneE" ]; then
    if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
      time python -m capreolus.run rerank.$step with searcher=qrels reranker=KNRM collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut reranker.extractor.document_cut=$doccut fold=s$FOLDNUM ;
    fi
  fi
fi
conda deactivate

