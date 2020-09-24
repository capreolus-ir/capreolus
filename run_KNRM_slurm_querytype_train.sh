#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
entitystrategy=noneE

assessed_set=random20
echo "sbatch --gres gpu:1  -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch --gres gpu:1  -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $entitystrategy $assessed_set;

./waitgpu.sh;

echo "sbatch -a 601-1200 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch  -a 601-1200 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $entitystrategy $assessed_set;

./waitgpu.sh;


assessed_set=top10
echo "sbatch --gres gpu:1  -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch  --gres gpu:1  -a 1-600 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $entitystrategy $assessed_set;
./waitgpu.sh;

echo "sbatch --gres gpu:1  -a 601-1200 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append run_KNRM_train_single.sh $domain $pipeline $entitystrategy $assessed_set;"
sbatch --gres gpu:1  -a 601-1200 -p gpu20 --mem-per-cpu=64G -o ${logfolder}train_gpu_KNRM_${domain}_${pipeline}_${assessed_set}.log --open-mode=append  run_KNRM_train_single.sh  $domain $pipeline $entitystrategy $assessed_set;

./waitgpu.sh;

