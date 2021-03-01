#!/bin/bash

logfolder=`cat paths_env_vars/logfolderpath`
pipeline=ENTITY_CONCEPT_JOINT_LINKING

method=$1
if [ "$method" == "" ];then
  echo "give input method LMD LMDEmb BM25c1.5 BM25cInf"
  exit
fi

MEM=32
SIMULRUN=10

if [ "$method" == "LMDEmb" ];then
  MEM=64
fi

declare -a domains=('book' 'food' 'travel')
declare -a dvtypes=("None" "all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
declare -a ents=('noneE' 'allE' 'domainE' 'onlyNE' 'domainOnlyNE')
if [ "$method" == "LMDEmb" ];then
  declare -a ents=('noneE')
fi

for entitystrategy in "${ents[@]}"
do
  for domainvocsp in "${dvtypes[@]}"
  do
    for domain in "${domains[@]}"
    do
      echo "sbatch -p cpu20 -c 2 -a 1-850%${SIMULRUN} --mem-per-cpu=32G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
      sbatch -p cpu20 -c 2 -a 1-850%${SIMULRUN} --mem-per-cpu=32G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
      sleep 5;
    done
    sleep 60;
  done
  sleep 3600; #wait an hour!
done
