cmd="filter=True searcher.name=BM25Postprocess 
	searcher.fields=query+question
	searcher.k1=0.5,0.6,0.7,0.8,0.9,1.0,1.1
	searcher.b=0.3,0.4,0.5,0.6
	benchmark.name=covid 
	benchmark.useprevqrels=False
	benchmark.udelqexpand=True
	benchmark.collection.round=2
	benchmark.collection.name=covid "

echo $cmd

if $1; then
	python run.py rank.search with $cmd 
fi

if $2; then
	python run.py rank.evaluate with $cmd 
fi
