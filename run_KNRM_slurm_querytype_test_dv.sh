#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING
entitystrategy=noneE

domain=$1
doccut=$2
echo "$domain - $doccut:"

assessed_set=random20
echo "sbatch --gres gpu:1 -a 1-1200%5 -p gpu20 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_test_single_dv.sh $domain $pipeline $entitystrategy $doccut $assessed_set;"
sbatch --gres gpu:1  -a 1-1200%5 -p gpu20 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append  run_KNRM_test_single_dv.sh  $domain $pipeline $entitystrategy $doccut $assessed_set;


#assessed_set=top10 train esham run nashode
#echo "sbatch --gres gpu:1  -a 1-1200%5 -p gpu20 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_test_single_dv.sh $domain $pipeline $entitystrategy $doccut $assessed_set;"
#sbatch --gres gpu:1  -a 1-1200%5 -p gpu20 --mem-per-cpu=64G -o ${logfolder}test_gpu_KNRM_${domain}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append  run_KNRM_test_single_dv.sh  $domain $pipeline $entitystrategy $doccut $assessed_set;


