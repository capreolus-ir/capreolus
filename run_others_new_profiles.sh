#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

domain=$1

#running with dv=None and filterq=None (first 10 of each 50s)
domainvocsp=None
entitystrategy=noneE

method=BM25c1.5
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 120;

method=BM25cInf
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 120;

method=LMDEmb
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}${method}_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
