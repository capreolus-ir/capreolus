#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING
entitystrategy=noneE
CPUNUM=2

domain=food
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy;

domain=book
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;

domain=movie
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;

domain=travel_wikivoyage
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=32G -o ${logfolder}LMDEmb_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMDEmb_single.sh $domain $pipeline $entitystrategy ;


