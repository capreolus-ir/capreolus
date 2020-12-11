#!/bin/bash
#SBATCH --job-name=msmarcopsg
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --account=your_slurm_account
#SBATCH --cpus-per-task=10
#SBATCH -o ./your_output_path

batch=16
itersize=30000
benchmark=msmarcopsg
searcher=msmarcopsgbm25
pretrained=bert-base-uncased # or /path/to/bert-base-uncased if the model folder is not moved to  `$HOME/setup_capr/src/capreolus`

source ~/.bashrc
conda activate caprolus
source /path/to/setup/dir/setup_capreolus_on_cc.bash  # e.g. $HOME/setup_capr/setup_capreolus_on_cc.bash

capreolus rerank.train with \
        benchmark.name=$benchmark \
        rank.searcher.name=$searcher \
        benchmark.collection.name=msmarcopsg \
        reranker.extractor.numpassages=1  \
        reranker.extractor.maxseqlen=512 \
        reranker.extractor.maxqlen=50 \
        reranker.name=TFBERTMaxP \
        reranker.pretrained=$pretrained \
        reranker.extractor.tokenizer.pretrained=$pretrained \
        reranker.extractor.usecache=True \
        reranker.trainer.usecache=True \
        reranker.trainer.niters=1 \
        reranker.trainer.itersize=$itersize \
        reranker.trainer.batch=$batch \
        optimize=MRR@10 rank.optimize=MRR@10