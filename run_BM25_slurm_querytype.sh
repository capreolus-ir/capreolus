#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_15092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j_%a.log run_BM25_single.sh $domain $pipeline $querytype $SLURM_ARRAY_TASK_ID $entitystrategy ;

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_%j_%a.log run_BM25_single_dv.sh $domain $pipeline $querytype $SLURM_ARRAY_TASK_ID $entitystrategy $domainvocsp;
  sleep 60;
done
