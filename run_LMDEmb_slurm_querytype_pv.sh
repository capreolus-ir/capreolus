#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_15092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=4

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
declare -a pvtypes=("topic-alltopics_tf_k-1" "topic-amazon_tf_k-1" "user-allusers_tf_k-1")

for filterq in "${pvtypes[@]}"
do
  sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_LMDEmb_single_pv.sh $domain $pipeline $querytype $SLURM_ARRAY_TASK_ID $entitystrategy $filterq ;
  sleep 30
done

for filterq in "${pvtypes[@]}"
do
  for domainvocsp in "${dstypes[@]}"
  do
    sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j_%a.log run_LMDEmb_single_dv_pv.sh $domain $pipeline $querytype $SLURM_ARRAY_TASK_ID $entitystrategy $domainvocsp $filterq ;
    sleep 30
  done
  sleep 30
done
