export JAVA_HOME=`cat paths_env_vars/javahomepath`
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=`cat paths_env_vars/capreolusresultpath` ;
export CAPREOLUS_CACHE=`cat paths_env_vars/capreoluscachepath` ;
export PYTHONPATH=`cat paths_env_vars/capreoluspythonpath` ;

pipeline=ENTITY_CONCEPT_JOINT_LINKING
dataset=kitt
declare -a profiles=('query' 'basicprofileMR' 'chatprofileMR' 'basicprofile_general' 'basicprofile_book' 'basicprofile_book_general' 'chatprofile_book' 'chatprofile_book_general')
domain=book

FOLDNUM=1

for querytype in "${profiles[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM;
done

for querytype in "${profiles[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline fold=s$FOLDNUM &
done
wait
echo "ALL finished"

for querytype in "${profiles[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc"  fold=s$FOLDNUM &
done
wait
echo "DOMAIN finished"

for querytype in "${profiles[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
done
wait
echo "onlyNE finished"

for querytype in "${profiles[@]}"
do
  time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True fold=s$FOLDNUM &
done
wait
echo "domainNE finished"