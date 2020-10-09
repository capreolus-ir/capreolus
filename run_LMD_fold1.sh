export JAVA_HOME=`cat paths_env_vars/javahomepath`
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=`cat paths_env_vars/capreolusresultpath` ;
export CAPREOLUS_CACHE=`cat paths_env_vars/capreoluscachepath` ;
export PYTHONPATH=`cat paths_env_vars/capreoluspythonpath` ;

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
pipeline=ENTITY_CONCEPT_JOINT_LINKING
dataset=kitt
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_food' 'basicprofile_travel' 'basicprofile_book' 'basicprofile_movie' 'basicprofile_food_general' 'basicprofile_travel_general' 'basicprofile_book_general' 'basicprofile_movie_general' 'chatprofile_general' 'chatprofile_food' 'chatprofile_travel' 'chatprofile_book' 'chatprofile_movie' 'chatprofile_food_general' 'chatprofile_travel_general' 'chatprofile_book_general' 'chatprofile_movie_general')
FOLDNUM=1

#to make the collection:
querytype=query
for domain in "${domains[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
  sleep 1
done
wait

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: none  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
    sleep 1
  done
  sleep
done
wait
echo "None finished"

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: all  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
    sleep 1
  done
done
wait
echo "ALL finished"

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: domain  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  fold=s$FOLDNUM &
    sleep 1
  done
done
wait
echo "DOMAIN finished"

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: onlyNE  "
  for domain in "${domains[@]}"
  do
     time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
     sleep 1
  done
done
wait
echo "onlyNE finished"

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: domNE  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
    sleep 1
  done
done
wait
echo "domainNE finished"