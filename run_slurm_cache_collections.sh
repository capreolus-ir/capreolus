#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING
querytype=query
entitystrategy=noneE
CPUNUM=2

domain=food
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;

domain=book
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;

domain=movie
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;

domain=travel_wikivoyage
sbatch -a 1-1 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single.sh $domain $pipeline $querytype $entitystrategy ;


