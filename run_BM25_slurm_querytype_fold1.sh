#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_15092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

sbatch -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log --wait export SLURM_ARRAY_TASK_ID=1 run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sbatch -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j.log export SLURM_ARRAY_TASK_ID=1 run_BM25_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp;
  sleep 30
done
