#!/bin/bash

#declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')

pipeline=ENTITY_CONCEPT_JOINT_LINKING
domain=$1
#for domain in "${domains[@]}"
#do
echo "source run_KNRM_slurm_querytype_train_dv.sh $domain $pipeline ;"
source run_KNRM_slurm_querytype_train_dv.sh $domain $pipeline ;
sleep 5
source run_KNRM_slurm_querytype_train_dv_p2.sh $domain $pipeline ;
#done




