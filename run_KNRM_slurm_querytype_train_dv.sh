#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
entitystrategy=noneE

declare -a doccuttypes=("most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df")

assessed_set=random20

for doccut in "${doccuttypes[@]}"
do
  echo "sbatch --gres gpu:1 -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_train_single_dv.sh $domain $pipeline $entitystrategy $doccut $assessed_set;"
  sbatch --gres gpu:1 -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single_dv.sh  $domain $pipeline $entitystrategy $doccut $assessed_set;
  ./waitgpu20.sh;
done



