# Setup Capreolus on Compute Canada

This page contains instructions to set up Capreolus on Compute Canada (CC).
Please follow [this guide](https://github.com/castorini/onboarding/blob/master/docs/cc-guide.md) to create an account on Compute Canada.

This instruction assume the users have anaconda or miniconda installed.

## Install Capreolus and its dependencies 
To setup, clone the repo and run the following scripts under the top-level capreolus: 
```bash
git clone https://github.com/capreolus-ir/capreolus && cd capreolus
# git checkout feature/msmarco_psg 

module load java/11
module load python/3.7
module load scipy-stack

ENVDIR=$HOME/venv/capreolus-env
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

# cat requirements.txt | xargs -n 1 pip install
pip install tf-models-official==2.5
cat requirements.txt | cut -d '#' -f 1  | grep "\S" | xargs -n 1 -i sh -c 'pip install --no-index {} || pip install {}'
pip install --no-index torch==1.9.0 spacy==2.2.2

# pip install tensorflow-text tf-models-official==2.5 tensorflow-ranking --no-deps
# pip install -r <( pip check | grep ^"tf-models-official\|tensorflow-ranking" | cut -d ' ' -f 4 | tr ',' ' ' | grep -v "tensorflow-text\|requirement" )
# pip install --no-index tensorflow==2.4.1
# pip install --no-index torch numpy==1.21.0 
# pip install --no-index torch spacy==2.2.2

# while read -r line; do echo $line | cut -d '#' -f 1 | grep "\S" |  xargs -n 1 pip install --no-index || pip install; done < requirements.txt
# while read -r line; do echo $line | cut -d '#' -f 1 | grep "\S" |  xargs -n 1 pip install; done < requirements.txt
# pip install tensorflow_ranking
# pip install profane>=0.2.0 pyjnius>=1.2.1 colorlog pyserini==0.12.0 ir_datasets gdown pymagnitude tensorflow_ranking pytrec_eval pytest-mock scispacy transformers 

# test

```

## Pre-download Huggingface models 
In case the server has no internet access, you can use the script `./scripts/download_model.sh` to pre-download huggingface models. 
`scripts/setup-cc.sh` pre-download `bert-base-uncased`, `bert-large-uncased` and `Capreolus/bert-base-msmarco` models under the setup directory. 
To download more, run `sh ./scripts/download_model.sh $model_name`. 
The model will then be downloaded to the current directory. 
You can then pass the model directory to Capreolus (e.g. `task.reranker.pretrained=/path/to/model task.reranker.extractor.tokenizer.pretrained=/path/to/model`).


## Pre-download MS MARCO Passage Dataset 
**After** specifying the `$CAPREOLUS_CACHE` and `$CAPREOLUS_RESULT` 
(For CC users, they should be set under `/scratch/your_user_name` since the cache and results can take a huge amount of space), 
run `sh download_data.sh` to pre-download the needed data for MS MARCO Passage dataset.
```bash
export CAPREOLUS_CACHE=/scratch/your_username/.capreolus/cache
export CAPREOLUS_RESULTS=/scratch/your_username/.capreolus/results
sh ./scripts/download_data.sh
``` 