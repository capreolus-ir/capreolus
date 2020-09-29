#!/bin/bash

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')

entitystrategy=$1
pipeline=ENTITY_CONCEPT_JOINT_LINKING

for domain in "${domains[@]}"
do
  echo "source run_LMDEmb_slurm_querytype.sh $domain $pipeline $entitystrategy &"
  source run_LMDEmb_slurm_querytype.sh $domain $pipeline $entitystrategy &
  sleep 120 
done




