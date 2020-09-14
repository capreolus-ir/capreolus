#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_14092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
declare -a pvtypes=("topic-alltopics_tf_k-1" "topic-amazon_tf_k-1" "user-allusers_tf_k-1")

for filterq in "${pvtypes[@]}"
do
  for FOLDNUM in {1..10};
  do
    sbatch -p cpu20 -c 10 --mem-per-cpu=32G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}%j.log run_LMDEmb_single_pv.sh $domain $pipeline $querytype $FOLDNUM $entitystrategy $filterq ;
    sleep 120;
  done
  sleep 300;
done

for filterq in "${pvtypes[@]}"
do
  for domainvocsp in "${dstypes[@]}"
  do
    for FOLDNUM in {1..10};
    do
      sbatch -p cpu20 -c 10 --mem-per-cpu=32G -o ${logfolder}BM25_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}%j.log run_LMDEmb_single_dv_pv.sh $domain $pipeline $querytype $FOLDNUM $entitystrategy $domainvocsp $filterq ;
      sleep 60;
    done
    sleep 300;
  done
  sleep 600;
done
