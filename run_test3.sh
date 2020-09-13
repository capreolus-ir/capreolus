export JAVA_HOME=/home/ghazaleh/Projects_Workspace_new/jdk/jdk-11.0.4
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=/GW/NeuralIR/nobackup/ghazaleh/results_test/ ;
export CAPREOLUS_CACHE=/GW/NeuralIR/nobackup/ghazaleh/cache_test/ ;
export PYTHONPATH=/GW/PKB/work/ghazaleh/capreolus/ ;

FOLDNUM=1;
domain=book;
dataset=kitt
pipeline=ENTITY_CONCEPT_JOINT_LINKING

querytype=basicprofile_book
time python -m capreolus.run rerank.evaluate with searcher=qrels reranker=LMDirichlet collection=$dataset collection.domain=$domain benchmark=$dataset benchmark.domain=$domain benchmark.querytype=$querytype reranker.extractor.entity_strategy=all reranker.extractor.entitylinking.pipeline=$pipeline reranker.extractor.filter_query=user-allusers_tf_k-1 fold=s$FOLDNUM &

wait
