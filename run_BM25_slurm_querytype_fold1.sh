#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=5

sbatch  -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_1_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;
