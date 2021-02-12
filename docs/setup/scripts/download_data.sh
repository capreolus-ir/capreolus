echo $CAPREOLUS_CACHE
if [ -z "$CAPREOLUS_CACHE" ]; then
        CAPREOLUS_CACHE="~/.capreolus/cache"
        echo "Warning: $CAPREOLUS_CACHE was not set, using the default path $CAPREOLUS_CACHE"
fi


# pre-download data
urls=("https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz" "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz")
dirs=("$CAPREOLUS_CACHE/collection-msmarcopsg/tmp" "$CAPREOLUS_CACHE/collection-msmarcopsg/benchmark-msmarcopsg/searcher-msmarcopsgbm25_b-0.4_fields-title_hits-1000_k1-0.9/tmp")

for i in $(seq 0 $((${#urls[@]}-1)))
do
        echo $i
        mkdir -p ${dirs[$i]}
        wget ${urls[$i]} -P ${dirs[$i]}
done