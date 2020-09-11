export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/NeuralIR/nobackup/ghazaleh/results_11092020/ ;
export CAPREOLUS_CACHE=/GW/NeuralIR/nobackup/ghazaleh/cache_11092020/ ;
export PYTHONPATH=/home/ghazaleh/Projects_Workspace_new/capreolus/ ;

domain=$1
pipeline=$2
echo $domain
echo $pipeline
dataset=kitt
declare -a basicprofiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_movie_general' 'basicprofile_book_general' 'basicprofile_movie_general')
declare -a chatprofiles=('chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  for querytype in "${basicprofiles[@]}"
  do
    echo "$querytype"
    echo "Entity: None "
    FOLDNUM=1
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM ;
    for FOLDNUM in {2..10};
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM &
    done
  done
fi
wait

if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  for querytype in "${chatprofiles[@]}"
  do
    echo "$querytype"
    echo "Entity: None "
    FOLDNUM=1
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM ;
    for FOLDNUM in {2..10};
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype fold=s$FOLDNUM &
    done
  done
fi
wait

echo "domain-specific-running"
declare -a dstypes=("all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")
for domainvocsp in "${dstypes[@]}"
do
  if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
    for querytype in "${basicprofiles[@]}"
    do
      echo "$querytype"
      echo "Entity: None $domainvocsp"
      FOLDNUM=1
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
      for FOLDNUM in {2..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
      done
    done
  fi
  wait
  if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
    for querytype in "${chatprofiles[@]}"
    do
      echo "$querytype"
      echo "Entity: None $domainvocsp"
      FOLDNUM=1
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM ;
      for FOLDNUM in {2..10};
      do
        time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp fold=s$FOLDNUM &
      done
    done
  fi
  wait
done

echo "FINISHED"
