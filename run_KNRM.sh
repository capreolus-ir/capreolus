#!/bin/bash

logfolder=`cat paths_env_vars/logfolderpath`
pipeline=ENTITY_CONCEPT_JOINT_LINKING

entitystrategy=noneE

step=$1
if [ "$step" == "" ];then
  echo "give input step train or evaluate"
  exit
fi
assessed_set=$2
if [ "assessed_set" == "" ];then
  echo "assessed_set shoult be given: random20 top10"
  exit
fi
SIMULRUN= 10

declare -a domains=('book' 'food' 'travel')
declare -a doccuttypes=("None" "most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

for doccut in "${doccuttypes[@]}"
do
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 4 -a 1-1020%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
    sbatch -p cpu20 -c 4 -a 1-1020%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
    sleep 10
  done
done

declare -a doccuttypes=("None" "most_frequent")
domain=alldomains
for doccut in "${doccuttypes[@]}"
do
  echo "sbatch -p cpu20 -c 4 -a 1-1020%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
  sbatch -p cpu20 -c 4 -a 1-1020%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 10
done
