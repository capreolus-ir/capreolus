export JAVA_HOME=`cat paths_env_vars/javahomepath`
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=`cat paths_env_vars/capreolusresultpath` ;
export CAPREOLUS_CACHE=`cat paths_env_vars/capreoluscachepath` ;
export PYTHONPATH=`cat paths_env_vars/capreoluspythonpath` ;

pipeline=ENTITY_CONCEPT_JOINT_LINKING
dataset=kitt
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'chatprofile_general' 'basicprofile_travel' 'basicprofile_travel_general' 'chatprofile_travel' 'chatprofile_travel_general')
declare -a fq=('None' 'topic-amazon_tf_k-1' 'user-allusers_tf_k-1' 'user-amazon_tf_k-1' 'topic-alltopics_tf_k-1')
declare -a dvtypes=("None" "all_domains_tf_k-1" "all_domains_df_k-1" "amazon_tf_k-1" "amazon_df_k-1")

domain=travel
ranker="LMDirichlet"
FOLDNUM=7

for domainvocsp in "${dvtypes[@]}";
do
  for filterq in "${fq[@]}";
  do
    for querytype in "${profiles[@]}";
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.query_vocab_specific=$filterq fold=s$FOLDNUM ;
    done
  done
done
wait

for domainvocsp in "${dvtypes[@]}";
do
  for filterq in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.query_vocab_specific=$filterq fold=s$FOLDNUM ;
    done
  done
done
wait
echo "ALL finished"

for domainvocsp in "${dvtypes[@]}";
do
  for filterq in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.query_vocab_specific=$filterq fold=s$FOLDNUM ;
    done
  done
done
wait
echo "DOMAIN finished"

for domainvocsp in "${dvtypes[@]}";
do
  for filterq in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.query_vocab_specific=$filterq fold=s$FOLDNUM ;
    done
  done
done
wait
echo "onlyNE finished"

for domainvocsp in "${dvtypes[@]}";
do
  for filterq in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=domain reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.domainrelatedness.strategy_NE="${domain}_prCacc"  reranker.extractor.domainrelatedness.strategy_C="${domain}_prCacc"  reranker.extractor.domainrelatedness.domain_relatedness_threshold_NE="${domain}_prCacc" reranker.extractor.domainrelatedness.domain_relatedness_threshold_C="${domain}_prCacc" reranker.extractor.onlyNamedEntities=True reranker.extractor.domain_vocab_specific=$domainvocsp reranker.extractor.query_vocab_specific=$filterq fold=s$FOLDNUM ;
    done
  done
done
wait
echo "domainNE finished"

declare -a fq=('None' 'unique_most_frequent' 'unique_topic-alltopics' 'unique_topic-amazon' 'unique_user-allusers' 'unique_user-amazon')
declare -a doccuttypes=("None" "most_frequent" "all_domains_tf" "all_domains_df" "amazon_tf" "amazon_df" )

domain=travel
ranker="KNRM"
FOLDNUM=7

step=train
assessed_set=random20
for doccut in "${doccuttypes[@]}";
do
  for querycut in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.$step with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut reranker.extractor.document_cut=$doccut fold=s$FOLDNUM ;
    done
  done
done


assessed_set=top10
for doccut in "${doccuttypes[@]}";
do
  for querycut in "${fq[@]}";
  do
    for querytype in "${profiles[@]}"
    do
      time python -m capreolus.run rerank.$step with searcher=qrels reranker=$ranker collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype benchmark.assessed_set=$assessed_set reranker.extractor.query_cut=$querycut reranker.extractor.document_cut=$doccut fold=s$FOLDNUM ;
    done
  done
done