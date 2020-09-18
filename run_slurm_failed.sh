#!/bin/bash
source /GW/home-12/ghazaleh/anaconda3/bin/activate venc
which python

export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/NeuralIR/nobackup/ghazaleh/results_14092020/ ;
export CAPREOLUS_CACHE=/GW/NeuralIR/nobackup/ghazaleh/cache_14092020/ ;
export PYTHONPATH=/home/ghazaleh/Projects_Workspace_new/capreolus_dev/capreolus/ ;

logfolder=/GW/NeuralIR/nobackup/ghazaleh/logs_14092020/

#/GW/NeuralIR/nobackup/ghazaleh/logs_14092020/LMDEmb_8_food_basicprofile_food_ENTITY_CONCEPT_JOINT_LINKING_noneE653543.log:2020-09-14:
domain=food
pipeline=ENTITY_CONCEPT_JOINT_LINKING
querytype=basicprofile_food
FOLDNUM=8
entitystrategy=noneE
dataset=kitt
sbatch -p cpu20 -c 10 --mem-per-cpu=24G -o ${logfolder}LMDEmb_${FOLDNUM}_${domain}_${querytype}_${pipeline}_${entitystrategy}%j.log run_LMDEmb_single.sh $domain $pipeline $querytype $FOLDNUM $entitystrategy ;

conda deactivate
