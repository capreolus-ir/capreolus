#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_15092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_LMD_single.sh $domain $pipeline $querytype $entitystrategy ;"
sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_LMD_single.sh $domain $pipeline $querytype $entitystrategy ;

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sleep 30
  echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j.log run_LMD_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp;"
  sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j.log run_LMD_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp;
done
