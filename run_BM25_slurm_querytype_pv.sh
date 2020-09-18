#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020/
domain=$1
pipeline=$2
querytype=$3
entitystrategy=$4
CPUNUM=2

declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
declare -a pvtypestopic=("topic-alltopics_tf_k-1" "topic-amazon_tf_k-1")
declare -a pvtypesuser=("user-allusers_tf_k-1")

bm25c=None
for filterq in "${pvtypestopic[@]}"
do
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    sleep 20
    echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;"
    sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;
  fi
done
for filterq in "${pvtypesuser[@]}"
do
  if [ "$querytype" != "query" ]; then
    sleep 20
    echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;"
    sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;
  fi
done

for filterq in "${pvtypestopic[@]}"
do
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    for domainvocsp in "${dstypes[@]}"
    do
      sleep 20
      echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;"
      sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;
    done
  fi
done
for filterq in "${pvtypesuser[@]}"
do
  if [ "$querytype" != "query" ]; then
    for domainvocsp in "${dstypes[@]}"
    do
      sleep 20
      echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;"
      sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;
    done
  fi
done


bm25c=1.5
for filterq in "${pvtypestopic[@]}"
do
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    sleep 20
    echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;"
    sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;
  fi
done
for filterq in "${pvtypesuser[@]}"
do
  if [ "$querytype" != "query" ]; then
    sleep 20
    echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;"
    sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${filterq}_%j.log run_BM25_single_pv.sh $domain $pipeline $querytype $entitystrategy $filterq $bm25c ;
  fi
done

for filterq in "${pvtypestopic[@]}"
do
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    for domainvocsp in "${dstypes[@]}"
    do
      sleep 20
      echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;"
      sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;
    done
  fi
done
for filterq in "${pvtypesuser[@]}"
do
  if [ "$querytype" != "query" ]; then
    for domainvocsp in "${dstypes[@]}"
    do
      sleep 20
      echo "sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;"
      sbatch -a 1-10 -p cpu20 -c $CPUNUM --mem-per-cpu=24G -o ${logfolder}BM25_%a_${domain}_${querytype}_${pipeline}_${entitystrategy}_${domainvocsp}_${filterq}_%j.log run_BM25_single_dv_pv.sh $domain $pipeline $querytype $entitystrategy $domainvocsp $filterq $bm25c ;
    done
  fi
done