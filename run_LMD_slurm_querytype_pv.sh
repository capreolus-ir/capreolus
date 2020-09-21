#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")

echo "sbatch -a 1-40 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_pv.log --open-mode=append run_LMD_single_pv.sh $domain $pipeline $querytype $entitystrategy ;"
sbatch -a 1-40 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_pv.log --open-mode=append run_LMD_single_pv.sh $domain $pipeline $querytype $entitystrategy ;

for domainvocsp in "${dstypes[@]}"
do
  sleep 10
  echo "sbatch -a 1-40 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}.log --open-mode=append run_LMD_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp ;"
  sbatch -a 1-40 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}.log --open-mode=append run_LMD_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp ;
done