#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
entitystrategy=$3
CPUNUM=4

echo "sbatch -a 1-200 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;"
sbatch -a 1-200 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;
./wait.sh;
declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  echo "sbatch -a 1-200 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_LMDEmb_single_dv.sh $domain $pipeline $entitystrategy $domainvocsp;"
  sbatch -a 1-200 -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_LMDEmb_single_dv.sh $domain $pipeline $entitystrategy $domainvocsp;
  ./wait.sh;
done
