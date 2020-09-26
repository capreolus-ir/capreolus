export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/GW/PKB/work/ghazaleh/capreolus/ ;

domain=$1
pipeline=ENTITY_CONCEPT_JOINT_LINKING
echo $domain
echo $pipeline
dataset=kitt
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')
declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")

if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  for querytype in "${profiles[@]}"
  do
    echo "$querytype"
    echo "Entity: None "
    for FOLDNUM in {1..10};
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM &
    done
    wait
  done

  for domainvocsp in "${dstypes[@]}"
  do
    echo $domainvocsp
    for querytype in "${profiles[@]}"
    do
      for FOLDNUM in {1..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
      done
      wait
    done
  done

fi
wait
if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  usp=user-allusers_tf_k-1
  for querytype in "${profiles[@]}"
  do
    if [ "$querytype" != "query" ]; then
      echo "$querytype"
      echo "Entity: None  Filter=user-allusers_tf_k-1"
      for FOLDNUM in {1..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings  collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp fold=s$FOLDNUM &
      done
    fi
    wait
  done

  for domainvocsp in "${dstypes[@]}"
  do
    echo $domainvocsp
    for querytype in "${profiles[@]}"
    do
      if [ "$querytype" != "query" ]; then
        for FOLDNUM in {1..10};
        do
          time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
        done
      fi
      wait
    done
  done
fi
wait
if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  usp=user-amazon_tf_k-1
  for querytype in "${profiles[@]}"
  do
    if [ "$querytype" != "query" ]; then
      echo "$querytype"
      echo "Entity: None  Filter=user-amazon_tf_k-1"
      for FOLDNUM in {1..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp fold=s$FOLDNUM &
      done
    fi
    wait
  done

  for domainvocsp in "${dstypes[@]}"
  do
    echo $domainvocsp
    for querytype in "${profiles[@]}"
    do
      if [ "$querytype" != "query" ]; then
        for FOLDNUM in {1..10};
        do
          time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
        done
      fi
      wait
    done
  done
fi
wait
if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  usp=topic-alltopics_tf_k-1
  for querytype in "${profiles[@]}"
  do
    if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
      echo "$querytype"
      echo "Entity: None  Filter=topic-alltopics_tf_k-1"
      for FOLDNUM in {1..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp fold=s$FOLDNUM &
      done
    fi
    wait
  done

  for domainvocsp in "${dstypes[@]}"
  do
    echo $domainvocsp
    for querytype in "${profiles[@]}"
    do
      if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
        for FOLDNUM in {1..10};
        do
          time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
        done
      fi
      wait
    done
  done
fi
wait
if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  usp=topic-amazon_tf_k-1
  for querytype in "${profiles[@]}"
  do
    if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
      echo "$querytype"
      echo "Entity: None  Filter=topic-amazon_tf_k-1"
      for FOLDNUM in {2..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp fold=s$FOLDNUM &
      done
    fi
    wait
  done

  for domainvocsp in "${dstypes[@]}"
  do
    echo $domainvocsp
    for querytype in "${profiles[@]}"
    do
      if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
        for FOLDNUM in {1..10};
        do
          time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichletWordEmbeddings collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.filter_query=$usp reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
        done
      fi
      wait
    done
  done

fi
echo "FINISHED"
