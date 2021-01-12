# Setup Capreolus on Compute Canada

This page contains instructions to set up Capreolus on Compute Canada (CC).
Please follow [this guide](https://github.com/castorini/onboarding/blob/master/docs/cc-guide.md) to create an account on Compute Canada.

This instruction assume the users have anaconda or miniconda installed.

## Install Capreolus and its dependencies 
To setup, clone the repo and run the following scripts under the top-level capreolus: 
```
git clone https://github.com/capreolus-ir/capreolus && cd capreolus
git checkout feature/msmarco_psg 
cd docs/setup

setup_dir="$HOME/setup_capr"  # don't remove this directory
mkdir -p $setup_dir
sh ./scripts/setup-cc.sh $setup_dir && cd ../..
source $setup_dir/setup_capreolus_on_cc.bash  # this needs to be run each time a new shell is created
python -m capreolus.run rank.print_config  # to check if the set-up is correct 
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
```
export CAPREOLUS_CACHE=/scratch/your_username/.capreolus/cache
export CAPREOLUS_RESULTS=/scratch/your_username/.capreolus/results
sh ./scripts/download_data.sh
``` 

If you are using Slurm, a sample shell script is presented in `./scripts/sample_slurm_script.sh`. 
