#!/bin/bash

logfolder=`cat paths_env_vars/logfolderpath`
pipeline=ENTITY_CONCEPT_JOINT_LINKING

method=LMD
if [ "$method" == "" ];then
  echo "give input method LMD LMDEmb BM25c1.5 BM25cInf"
  exit
fi

SIMULRUN=10
STAGE=$1
if [ "$STAGE" == "" ];then
  echo "give input stage collection entities similarities"
  exit
fi
declare -a domains=('book' 'food' 'travel')
declare -a dvtypes=("None" "all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
domainvocsp=None

if [ "$STAGE" == "collection" ];then
  # first, only one fold per domain to create the collections:
  entitystrategy=noneE
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 2 -a 1-850:850%1 --mem-per-cpu=32G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
    sbatch -p cpu20 -c 2 -a 1-850:850%1 --mem-per-cpu=32G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  done
  # download the word vector:
  domain=book
  method=LMDEmb
  echo "sbatch -p cpu20 -c 2 -a 1-850:850%1 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
  sbatch -p cpu20 -c 2 -a 1-850:850%1 --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;

  # download w2v of the knrm
  step=train
  doccut=None
  assessed_set=random20
  echo "sbatch -p cpu20 -c 4 -a 1-1020:1020%1 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;"
  sbatch -p cpu20 -c 4 -a 1-1020:1020%1 --mem-per-cpu=64G -o ${logfolder}${step}_KNRM_${domain}_${entitystrategy}_${pipeline}_${doccut}_${assessed_set}.log --open-mode=append run_KNRM_single_complete.sh  $domain $pipeline $entitystrategy $doccut $assessed_set $step;
fi

if [ "$STAGE" == "entities" ];then
  # entity extraction per profiletype
  entitystrategy=allE
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 2 -a 1-850:50%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
    sbatch -p cpu20 -c 2 -a 1-850:50%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  done
fi


if [ "$STAGE" == "similarities" ];then
  # calculating entity similarities per profiletype
  entitystrategy=domainE
  for domain in "${domains[@]}"
  do
    echo "sbatch -p cpu20 -c 2 -a 1-850:50%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;"
    sbatch -p cpu20 -c 2 -a 1-850:50%${SIMULRUN} --mem-per-cpu=64G -o ${logfolder}${method}_${domain}_${entitystrategy}_${pipeline}_${domainvocsp}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $entitystrategy $domainvocsp $method;
  done
fi

# since we are now only writing to most of the debug caches (except for entities and similarities), it's okay to just run others simul