#!/bin/bash
#SBATCH --job-name=msmarcopsg
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --account=your_slurm_account
#SBATCH --cpus-per-task=10
#SBATCH -o ./output.log

lr=1e-3
bertlr=2e-5
batch_size=16
niters=10
warmupiters=1
decayiters=$niters  # either same with $itersize or 0

python -m capreolus.run rerank.train with \
    file=docs/reproduction/config_msmarco.txt  \
    reranker.trainer.batch=$batch_size \
    reranker.trainer.lr=$lr \
    reranker.trainer.bertlr=$bertlr \
    reranker.trainer.niters=$niters \
    reranker.trainer.warmupiters=$warmupiters \
    reranker.trainer.decayiters=$decayiters \
    fold=s1
