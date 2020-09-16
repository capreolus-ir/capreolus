#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_16092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=4

echo "sbatch -a 1-10 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_LMDEmb_single.sh $domain $pipeline $querytype $entitystrategy ;"
sbatch -a 1-10 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_LMDEmb_single.sh $domain $pipeline $querytype $entitystrategy ;

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sleep 30
  echo "sbatch -a 1-10 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j.log script_to_run_one_fold.sh $domain $pipeline $querytype $entitystrategy $domainvocsp;"
  sbatch -a 1-10 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j.log script_to_run_one_fold.sh $domain $pipeline $querytype $entitystrategy $domainvocsp;
done
