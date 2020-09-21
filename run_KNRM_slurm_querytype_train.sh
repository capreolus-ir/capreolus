#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=noneE

assessed_set=random20
querycut=None
echo "sbatch -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;"
sbatch  -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;
sleep 5
querycut=unique_most_frequent
echo "sbatch -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;"
sbatch  -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;

sleep 10
assessed_set=top10
querycut=None
echo "sbatch -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;"
sbatch  -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;
sleep 5
querycut=unique_most_frequent
echo "sbatch -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;"
sbatch  -a 1-10 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_%a_${domain}_${querytype}_${pipeline}_${querycut}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $querytype $entitystrategy $assessed_set $querycut;
