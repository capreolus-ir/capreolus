#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=alldomains
pipeline=ENTITY_CONCEPT_JOINT_LINKING
entitystrategy=noneE


declare -a doccuttypes=("most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df")

assessed_set=random20
echo "sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single_sorted.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single_sorted.sh  $domain $pipeline $entitystrategy $assessed_set;

sleep 120

assessed_set=top10
echo "sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single_sorted.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single_sorted.sh  $domain $pipeline $entitystrategy $assessed_set;

sleep 120

assessed_set=random20
for doccut in "${doccuttypes[@]}"
do
  echo "sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_train_single_sorted_dv.sh $domain $pipeline $entitystrategy $doccut $assessed_set;"
  sbatch -p cpu20 -c 5 -a 1-300 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single_sorted_dv.sh  $domain $pipeline $entitystrategy $doccut $assessed_set;
  sleep 120
done

