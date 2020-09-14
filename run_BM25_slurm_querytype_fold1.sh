#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_14092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4

sbatch -p cpu20 -c 10 --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}%j.log --wait run_BM25_single.sh $domain $pipeline $querytype 1 $entitystrategy ;

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sbatch -p cpu20 -c 10 --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}%j.log run_BM25_single_dv.sh $domain $pipeline $querytype 1 $entitystrategy $domainvocsp;
  sleep 60;
done
