export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/NeuralIR/nobackup/ghazaleh/results_02092020/ ;
export CAPREOLUS_CACHE=/GW/NeuralIR/nobackup/ghazaleh/cache_02092020/ ;
export PYTHONPATH=/home/ghazaleh/Projects_Workspace_new/capreolus/ ;

domain=$1
pipeline=$2
echo $domain
echo $pipeline
dataset=kitt
declare -a arr=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book_movie' 'basicprofile_book' 'basicprofile_movie' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_hobbies')

for querytype in "${arr[@]}"
do
  echo "$querytype"
  echo "Entity: all "
  FOLDNUM=1
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM ;
  for FOLDNUM in {2..10};
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
  done
  wait
done
wait

for querytype in "${arr[@]}"
do
  if [ "$querytype" != "query" ]; then
    echo "$querytype"
    echo "Entity: all  Filter=user-k-1"
    FOLDNUM=1
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=user_specific_k-1 fold=s$FOLDNUM ;
    for FOLDNUM in {2..10};
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=user_specific_k-1 fold=s$FOLDNUM &
    done
    wait
  fi
done
wait

for querytype in "${arr[@]}"
do
  if [ "$querytype" != "query" ] && [ "$querytype" != "basicprofile" ] && [ "$querytype" != "chatprofile" ]; then
    echo "$querytype"
    echo "Entity: all  Filter=domain-k-1"
    FOLDNUM=1
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=domain_specific_k-1 fold=s$FOLDNUM ;
    for FOLDNUM in {2..10};
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=BM25 reranker.b=0.75 reranker.k1=1.5 collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=domain_specific_k-1 fold=s$FOLDNUM &
    done
    wait
  fi
done
wait
echo "FINISHED"
