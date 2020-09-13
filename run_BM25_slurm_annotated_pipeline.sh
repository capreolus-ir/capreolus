#!/bin/bash

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

entitystrategy=$1
pipeline=ENTITY_CONCEPT_JOINT_LINKING

for domain in "${domains[@]}"
do
  for querytype in "${profiles[@]}"
  do
./run_BM25_slurm_querytype.sh $domain $pipeline $querytype $entitystrategy
  done
done
