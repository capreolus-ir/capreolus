export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/D5data-13/ghazaleh/ranking_outputs/results_18092020/ ;
export CAPREOLUS_CACHE=/GW/D5data-13/ghazaleh/ranking_outputs/cache_18092020/ ;
export PYTHONPATH=/GW/PKB/work/ghazaleh/capreolus/ ;

declare -a domains=('book' 'food' 'movie' 'travel_wikivoyage')
pipeline=ENTITY_CONCEPT_JOINT_LINKING
dataset=kitt
declare -a profiles=('chatprofile_food_general' 'chatprofile_travel_general' 'chatprofile_book_general' 'chatprofile_movie_general' 'chatprofile_hobbies_general')
FOLDNUM=1

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: none  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
  done
done

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: all  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
  done
done
wait
echo "None and ALL finished"

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: domain  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  fold=s$FOLDNUM &
  done
done
wait

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: onlyNE  "
  for domain in "${domains[@]}"
  do
     time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
  done
done
wait

for querytype in "${profiles[@]}"
do
  echo "$querytype"
  echo "Entity: domNE  "
  for domain in "${domains[@]}"
  do
    time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
  done
done
wait
