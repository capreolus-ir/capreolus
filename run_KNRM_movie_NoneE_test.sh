#!/bin/bash
export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/home/ghazaleh/capreolus_dev/capreolus/ ;

domain=movie
pipeline=ENTITY_CONCEPT_JOINT_LINKING
dataset=kitt
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'chatprofile_book' 'chatprofile_movie' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_food_general' 'basicprofile_travel_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_hobbies')
declare -a doccuttypes=("most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )
declare -a qcuttypes=("None" "unique_most_frequent" "unique_topic-alltopics" "unique_topic-amazon" "unique_user-allusers" "unique_user-amazon" )
assessed_set=random20

if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  for querytype in "${profiles[@]}"
  do
    echo "$querytype"
    echo "Entity: None "
    for doccut in "${doccuttypes[@]}"
    do
      for querycut in "${qcuttypes[@]}"
      do
        for FOLDNUM in {1..10};
        do
          time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=KNRM collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut reranker.extractor.document_cut=$doccut fold=s$FOLDNUM &
        done
      done
      wait
    done
  done
fi
