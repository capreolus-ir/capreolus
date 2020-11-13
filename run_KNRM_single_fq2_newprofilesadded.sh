#!/bin/bash
source `cat paths_env_vars/virtualenv`
which python

export JAVA_HOME=`cat paths_env_vars/javahomepath`
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=`cat paths_env_vars/capreolusresultpath` ;
export CAPREOLUS_CACHE=`cat paths_env_vars/capreoluscachepath` ;
export PYTHONPATH=`cat paths_env_vars/capreoluspythonpath` ;

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

