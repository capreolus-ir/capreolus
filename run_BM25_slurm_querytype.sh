#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

bm25c=None
echo "sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;"
sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;

sleep 20

bm25c=1.5
echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;"
sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;

bm25c=None
declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sleep 20
  echo "sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_BM25_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $bm25c;"
  sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_BM25_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $bm25c;
done

sleep 20

bm25c=1.5
declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  sleep 20
  echo "sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_BM25_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $bm25c;"
  sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}.log --open-mode=append run_BM25_single_dv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $bm25c;
done