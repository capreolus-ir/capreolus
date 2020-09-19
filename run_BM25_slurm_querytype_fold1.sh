#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=5

bm25c=None
echo "sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;"
sbatch  -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}.log --open-mode=append run_BM25_single.sh $domain $pipeline $querytype $entitystrategy $bm25c;
