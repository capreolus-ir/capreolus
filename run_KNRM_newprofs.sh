#!/bin/bash

logfolder=`cat paths_env_vars/logfolderpath`
pipeline=ENTITY_CONCEPT_JOINT_LINKING

entitystrategy=noneE

step=$1
if [ "$step" == "" ];then
  echo "give input step train or evaluate"
  exit
fi

declare -a domains=('book' 'food' 'travel_wikivoyage')
declare -a doccuttypes=("None" "most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

assessed_set=random20
for doccut in "${doccuttypes[@]}"
do
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 4 -a 1-40 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
    sbatch -p cpu20 -c 4 -a 1-40 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
    sleep 30;
  done
done
#sleep 300;
#
#assessed_set=top10
#for doccut in "${doccuttypes[@]}"
#do
#  for domain in "${domains[@]}"
#  do
#    echo "sbatch -p cpu20 -c 4 -a 1-40 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
#    sbatch -p cpu20 -c 4 -a 1-40 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_fq2_newprofilesadded.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
#    sleep 30;
#  done
#done

