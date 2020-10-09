#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_30092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

entitystrategy=noneE

step=$1
if [ "$step" == "" ];then
  echo "give input step train or evaluate"
  exit
fi

domain=alldomains
declare -a doccuttypes=("None" "most_frequent")

assessed_set=random20
for doccut in "${doccuttypes[@]}"
do
    echo "sbatch -p cpu20 -c 4 -a 1-100%20 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
    sbatch -p cpu20 -c 4 -a 1-100%20 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
    sleep 60;
done
sleep 60;
assessed_set=top10
for doccut in "${doccuttypes[@]}"
do
  echo "sbatch -p cpu20 -c 4 -a 1-100%20 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
  sbatch -p cpu20 -c 4 -a 1-100%20 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 60;
done

