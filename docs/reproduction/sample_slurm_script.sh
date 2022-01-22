#!/bin/bash
#SBATCH --job-name=msmarcopsg
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --time=72:00:00
#SBATCH --account=your_slurm_account
#SBATCH --cpus-per-task=16

#SBATCH -o ./msmarco-psg-output.log


# Modify the following lines according to your setup process
module load arch/avx512 StdEnv/2020 java/11 python/3.7 scipy-stack
ENVDIR=$HOME/venv/capreolus-env
source $ENVDIR/bin/activate

niters=10
batch_size=16
validatefreq=$niters # to ensure the validation is run only at the end of training
decayiters=$niters   # either same with $itersize or 0
threshold=1000       # the top-k documents to rerank

python -m capreolus.run rerank.train with \
	file=docs/reproduction/config_msmarco.txt  \
	threshold=$threshold \
	reranker.trainer.niters=$niters \
	reranker.trainer.batch=$batch_size \
	reranker.trainer.decayiters=$decayiters \
	reranker.trainer.validatefreq=$validatefreq \
	fold=s1
