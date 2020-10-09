#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_30092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

method=$1
if [ "$method" == "" ];then
  echo "give input method LMD LMDEmb BM25c1.5 BM25cInf"
  exit
fi

declare -a domains=('travel_wikivoyage')
declare -a dvtypes=("None" "all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")


entitystrategy=onlyNE
for domainvocsp in "${dvtypes[@]}"
do
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 2 -a 1-1050 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
    sbatch -p cpu20 -c 2 -a 1-1050 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
    sleep 60;
  done
done
