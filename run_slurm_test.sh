#!/bin/bash

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_15092020/
domain=food
pipeline=ENTITY_CONCEPT_JOINT_LINKING
querytype=chatprofile_hobbies
entitystrategy=domainOnlyNE
CPUNUM=2

sbatch -a 2-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;
