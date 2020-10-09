#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_30092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

method=LMDEmb
if [ "$method" == "" ];then
  echo "give input method LMD LMDEmb BM25c1.5 BM25cInf"
  exit
fi

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
declare -a dvtypes=("None" "all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")

entitystrategy=noneE
domainvocsp=None
domain=food
sbatch -p cpu20 -c 4 -a 53-60%2 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
#sleep 100;

#domain=movie
#sbatch -p cpu20 -c 2 -a 121-130%5 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method; 
#sleep 10;
#sbatch -p cpu20 -c 2 -a 151-160%5 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
#sleep 10;
#sbatch -p cpu20 -c 2 -a 151-160%5 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
#sleep 10;

#domain=travel_wikivoyage
#sbatch -p cpu20 -c 2 -a 121-130%5 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
#sleep 100;

#domain=food
#sbatch -p cpu20 -c 2 -a 61-70%5 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

