#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING
entitystrategy=noneE

domain=$1
echo "$domain:"

declare -a doccuttypes=("most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

assessed_set=random20

for doccut in "${doccuttypes[@]}"
do
  echo "sbatch -p cpu20 -c 2 -a 1-1200 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_test_single_dv.sh $domain $pipeline $entitystrategy $doccut $assessed_set;"
  sbatch -p cpu20 -c 2 -a 1-1200 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append  run_KNRM_test_single_dv.sh  $domain $pipeline $entitystrategy $doccut $assessed_set;
  sleep 30;
done
