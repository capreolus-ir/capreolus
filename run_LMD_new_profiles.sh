#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

domain=$1

#running with dv=None and filterq=None (first 10 of each 50s)
domainvocsp=None
method=LMD

entitystrategy=noneE
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

sleep 600;
entitystrategy=allE
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

sleep 600;
entitystrategy=domainE
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

sleep 600;
entitystrategy=onlyNE
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

sleep 600;
entitystrategy=domainOnlyNE
sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
sleep 10;
sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_LMD_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
