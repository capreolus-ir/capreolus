#!/bin/bash

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')

pipeline=ENTITY_CONCEPT_JOINT_LINKING

for domain in "${domains[@]}"
do
    echo "source run_KNRM_slurm_querytype_train.sh $domain $pipeline $querytype &"
    source run_KNRM_slurm_querytype_train.sh $domain $pipeline $querytype &
    sleep 60
done




