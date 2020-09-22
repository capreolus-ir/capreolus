#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
entitystrategy=$3
CPUNUM=2

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")

echo "sbatch -a 1-800 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_${domain}_${pipeline}_${entitystrategy}_pv.log --open-mode=append run_LMD_single_pv.sh $domain $pipeline $entitystrategy ;"
sbatch -a 1-800 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_${domain}_${pipeline}_${entitystrategy}_pv.log --open-mode=append run_LMD_single_pv.sh $domain $pipeline $entitystrategy ;

./wait.sh;

for domainvocsp in "${dstypes[@]}"
do
  echo "sbatch -a 1-800 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_${domain}_${pipeline}_${entitystrategy}_${domainvocsp}_pv.log --open-mode=append run_LMD_single_dv_pv.sh $domain $pipeline $entitystrategy $domainvocsp ;"
  sbatch -a 1-800 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}LMD_${domain}_${pipeline}_${entitystrategy}_${domainvocsp}_pv.log --open-mode=append run_LMD_single_dv_pv.sh $domain $pipeline $entitystrategy $domainvocsp ;
  ./wait.sh;
done
