#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
pipeline=ENTITY_CONCEPT_JOINT_LINKING

#domain=$1

#running with dv=None and filterq=None (first 10 of each 50s)
domainvocsp=None
method=LMD
declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
for domain in "${domains[@]}"
do
  entitystrategy=noneE
  sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1251-1260 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1301-1310 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1351-1360 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1401-1410 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

  sleep 120;
  entitystrategy=allE
  sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1251-1260 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1301-1310 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1351-1360 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1401-1410 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

  sleep 120;
  entitystrategy=domainE
  sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1251-1260 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1301-1310 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1351-1360 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1401-1410 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

  sleep 120;
  entitystrategy=onlyNE
  sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1251-1260 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1301-1310 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1351-1360 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1401-1410 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

  sleep 120;
  entitystrategy=domainOnlyNE
  sbatch -p cpu20 -c 2 -a 1001-1010 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1051-1060 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1101-1110 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1151-1160 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1201-1210 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1251-1260 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1301-1310 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1351-1360 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  sleep 10;
  sbatch -p cpu20 -c 2 -a 1401-1410 --mem-per-cpu=64G -o ${logfolder}LMD_newprofiles_${domain}_${pipeline}_${entitystrategy}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
done