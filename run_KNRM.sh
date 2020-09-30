#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_30092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

entitystrategy=noneE

step=$1
if [ "$step" == "" ];then
  echo "give input step train or evaluate"
  exit
fi

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
declare -a doccuttypes=("most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

doccut=None
assessed_set=top10
for domain in "${domains[@]}"
do
  sbatch -p cpu20 -c 4 -a 1-1260 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 60;
done
assessed_set=random20
for domain in "${domains[@]}"
do
  sbatch -p cpu20 -c 4 -a 1-1260 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 60;
done
sleep 1000;

for doccut in "${doccuttypes[@]}"
do
  for domain in "${domains[@]}"
  do
    sbatch -p cpu20 -c 4 -a 1-1260 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
    sleep 300;
  done
done



assessed_set=random20
for doccut in "${doccuttypes[@]}"
do
  sbatch -p cpu20 -c 2 -a 1201-1440 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_newprofiles_${domain}_${pipeline}_${entitystrategy}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 60;
done
