#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
entitystrategy=$3
CPUNUM=4

sbatch --array [1,11,21,31,41,51,61,71,81,91,101,110,121,131,141,151,161,171,181,191] -t 2-00:00:00 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;
