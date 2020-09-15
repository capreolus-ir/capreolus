#!/bin/bash

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

entitystrategy=$1
pipeline=ENTITY_CONCEPT_JOINT_LINKING

for querytype in "${profiles[@]}"
do
  for domain in "${domains[@]}"
  do
    echo "source run_BM25_slurm_querytype_pv.sh $domain $pipeline $querytype $entitystrategy &"
    source run_BM25_slurm_querytype_pv.sh $domain $pipeline $querytype $entitystrategy &
    sleep 10
  done
done




