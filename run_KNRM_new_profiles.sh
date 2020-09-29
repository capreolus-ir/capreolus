#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

domain=$1
entitystrategy=noneE

step=evaluate

declare -a doccuttypes=("None" "most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

doccut=None
assessed_set=top10
sbatch -p cpu20 -c 2 -a 1201-1440 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_newprofiles_${domain}_${pipeline}_${entitystrategy}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
sleep 60;

assessed_set=random20
for doccut in "${doccuttypes[@]}"
do
  sbatch -p cpu20 -c 2 -a 1201-1440 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_newprofiles_${domain}_${pipeline}_${entitystrategy}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
  sleep 60;
done
