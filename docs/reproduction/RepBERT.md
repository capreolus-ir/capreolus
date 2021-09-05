# Capreolus: Ranking robust04 with RepBERT 
This page contains instructions for running RepBERT on the robust04 benchmark.

[RepBERT: Contextualized Text Embeddings for First-Stage Retrieval](https://arxiv.org/abs/2006.15498) by Zhan et al.

## Setup

1. Install Capreolus from the master branch - the changes required to make dense retrieval work are not yet in the pip version
2. Make sure additional dependencies like the `faiss-cpu` package are installed in your conda/virtual environment
3. Set the `CAPREOLUS_FAISS_LOG` environment variable to the file where you want to see logs that are specific to the dense ranker. Also set `CAPREOLUS_LOGGING` to `DEBUG`. For example:
```angular2html
 export CAPREOLUS_FAISS_LOG=$HOME/faiss_logs/denserank_train_yang19_finetune_s1.log
 export CAPREOLUS_LOGGING=DEBUG
```

## RepBERT as a first-stage dense retriever

Capreolus uses [faiss](https://github.com/facebookresearch/faiss) as a vector store for the encoded document representations that RepBERT creates. Since encoding an entire document collection is time-consuming, the new `denserank` task in capreolus can optionally "shard" the vector index creation step across multiple gpus/machines. Using RepBERT as a first-stage ranker (i.e as a replacement for BM25) requires 3 steps:

(Additional instructions for executing on the MPI-INF slurm cluster are provided at the very end of this document)

####1. Train RepBERT
```
python /home/kjose/capreolus/capreolus/run.py denserank.trainencoder \
 with \
 numshards=400 \
 benchmark.name=robust04.yang19 \
 encoder.extractor.usecache=True \
 encoder.name=repbert \
 encoder.trainer.lr=0.000003 \
 fold=s1 \ 
 encoder.trainer.validatefreq=1 \
 encoder.trainer.niters=30 \
 encoder.trainer.batch=16 \
 encoder.sampler.name=pair \
 encoder.trainer.gradacc=1
```
Notice the new `numshards` parameter that is specific to the `denserank` task as well as the `encoder` parameter that indicates that we are training an encoder (i.e a dense retrieval model) and not a re-ranker. The `numshards` parameter is arbitrary, and is usually determined by the number of GPUs/workers available for you. On the MPI-INF slurm cluster (150 GPUs) encoding the __entire__ Robust04 collection using RepBERT finishes under 10 minutes if `numshards` is set to 400. For larger collections (eg: GOV2), setting `numshards` as `5000` ensures that the entire collection is indexed in unders two hours. One heuristic is to choose a `numshards` sufficiently high so that each shard is small enough (because higher the `numshards`, the smaller the number of documents stored in a single shard) to be created under 30 minutes on a single GPU. You might also want to train (different) dense rankers for each of the folds (i.e set `fold=s2`, `fold=s3` e.t.c)

N.B: The `repbert` encoder that we use here is initialzed using hugging face's `bert-base-uncased` weights. Use the `repbertpretrained` encoder to use the checkpoint that the authors released that was fine-tuned on the MSMarco dataset.

####2. Create a FAISS index 

In this step capreolus uses the encoder trained in the previous step to encode the entire document collection. Make sure that the parameters in this step are identical to the one you used to train the encoder - otherwise capreolus won't be able to find the cached encoder.

```angular2html
python /home/kjose/capreolus/capreolus/run.py denserank.createshard with \
numshards=400 \
shard=$SHARD_ID \
benchmark.name=robust04.yang19 \
encoder.extractor.usecache=True \
encoder.name=repbert \
encoder.trainer.lr=0.000003 \
fold=s1 \
encoder.trainer.validatefreq=1 \
encoder.trainer.niters=30 \
encoder.trainer.batch=16 \
encoder.sampler.name=pair \
encoder.trainer.gradacc=1
```

The only extra parameter here is `shard` - this is used to indicate which one of the `400` shards should be created. To index the entire document collection, it is necessary to execute this command `numshard` times (`400` times in out example), each time with a different `$SHARD_ID` environment variable


####3. Perform a dense retrieval (i.e evaluate) 
The next step is to encode every query in the test set and use it to retrieve documents from the FAISS index (which was created using the trained encoder) using similarity search.

```
python /home/kjose/capreolus/capreolus/run.py denserank.evaluate with \
 numshards=400 \
 benchmark.name=robust04.yang19 \
 encoder.extractor.usecache=True \
 encoder.name=repbert \
 encoder.trainer.lr=0.000003 \
 fold=s1 \
 encoder.trainer.validatefreq=1 \
 encoder.trainer.niters=30 \
 encoder.trainer.batch=16 \
 encoder.sampler.name=pair \
 encoder.trainer.gradacc=1
```
The rank metrics will be written to the file set in `$CAPREOLUS_FAISS_LOG`

### Instructions for MPI-INF slurm

The following scripts can be used to train and evaluate RepBERT for all 5 folds of Robust04 on MPI-INF's slurm cluster. These scripts still need to be run in two phases - one to train the encoder and create the FAISS indexes, and another script to evaluate the encoders. Calculating cross-validated metrics is not supported yet - the user will have to manually average the metrics obtained for each fold.


#### 1. To Train the encoder and create the FAISS shards:
```
sbatch repbert-yang19-finetune.sh
```
The actual contents of `repbert-yang19-finetune.sh` are described further down this document 

#### 2. To Evaluate the encoder:
This command needs to be executed after the previous one has successfully completed. Evaluate the logs to make sure that the previous step did indeed complete successfully.

```
sbatch repbert-yang19-finetune-eval.sh
```

TODO: Make the second command (i.e the one for eval) depend on the first sbatch command so that the user can queue up an experiment and get the results the next day, instead of having to manually run a separate eval script.

#### You need to create the following files 

##### repbert-yang19-finetune.sh
```

RES=$(sbatch --parsable --export=ALL,fold=s1 repbert-yang19-finetune-train.sh)
echo $RES
RES2=$(sbatch --parsable --dependency=afterok:${RES} --export=ALL,fold=s1 -a 1-400 repbert-yang19-finetune-shard.sh)
echo $RES2
RES=$(sbatch --parsable --export=ALL,fold=s2 repbert-yang19-finetune-train.sh)
echo $RES
RES2=$(sbatch --parsable --dependency=afterok:${RES} --export=ALL,fold=s2 -a 1-400 repbert-yang19-finetune-shard.sh)
echo $RES2
RES=$(sbatch --parsable --export=ALL,fold=s3 repbert-yang19-finetune-train.sh)
echo $RES
RES2=$(sbatch --parsable --dependency=afterok:${RES} --export=ALL,fold=s3 -a 1-400 repbert-yang19-finetune-shard.sh)
echo $RES2
RES=$(sbatch --parsable --export=ALL,fold=s4 repbert-yang19-finetune-train.sh)
echo $RES
RES2=$(sbatch --parsable --dependency=afterok:${RES} --export=ALL,fold=s4 -a 1-400 repbert-yang19-finetune-shard.sh)
echo $RES2
RES=$(sbatch --parsable --export=ALL,fold=s5 repbert-yang19-finetune-train.sh)
echo $RES
RES2=$(sbatch --parsable --dependency=afterok:${RES} --export=ALL,fold=s5 -a 1-400 repbert-yang19-finetune-shard.sh)
echo $RES2
```

##### repbert-yang19-finetune-train.sh
```
#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /home/kjose/slurm-scripts/outputs/denserank-train-yang19-finetune-all.log
#SBATCH -t 23:59:30

eval "$(conda shell.bash hook)"
conda activate capri
export JAVA_HOME=/home/kjose/jdk-11/
export CAPREOLUS_CACHE="/GW/NeuralIR/nobackup/kevin_cache"
export CAPREOLUS_RESULTS="/GW/NeuralIR/nobackup/kevin_results"
export PYTHONPATH=$PYTHONPATH:/home/kjose/capreolus
export PATH="/home/kjose/trec_eval:$PATH"
export CAPREOLUS_LOGGING=DEBUG
export CAPREOLUS_FAISS_LOG=/home/kjose/slurm-scripts/faiss_logs/faiss_denserank_train_yang19_finetune_${fold}.log

echo $CUDA_VISIBLE_DEVICES

python /home/kjose/capreolus/capreolus/run.py denserank.trainencoder with \
numshards=400 \
benchmark.name=robust04.yang19 \
encoder.extractor.usecache=True \
encoder.name=repbert \
encoder.trainer.lr=0.000003 \
fold=${fold} \
encoder.trainer.validatefreq=1 \
encoder.trainer.niters=30 \
encoder.trainer.batch=16 \
encoder.sampler.name=pair \
encoder.trainer.gradacc=1

```

##### repbert-yang19-finetune-shard.sh
```
#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /home/kjose/slurm-scripts/outputs/denserank-shard-yang19-finetune-all.log
#SBATCH -t 0:29:59

eval "$(conda shell.bash hook)"
conda activate capri
export JAVA_HOME=/home/kjose/jdk-11/
export CAPREOLUS_CACHE="/GW/NeuralIR/nobackup/kevin_cache"
export CAPREOLUS_RESULTS="/GW/NeuralIR/nobackup/kevin_results"
export PYTHONPATH=$PYTHONPATH:/home/kjose/capreolus
export PATH="/home/kjose/trec_eval:$PATH"
export CAPREOLUS_LOGGING=DEBUG
export CAPREOLUS_FAISS_LOG=/home/kjose/slurm-scripts/faiss_logs/faiss_denserank_shard_yang19_finetune_${fold}.log

echo $CUDA_VISIBLE_DEVICES

python /home/kjose/capreolus/capreolus/run.py denserank.createshard with \
numshards=400 \
shard=${SLURM_ARRAY_TASK_ID} \
benchmark.name=robust04.yang19 \
encoder.extractor.usecache=True \
encoder.name=repbert \
encoder.trainer.lr=0.000003 \
fold=${fold} \
encoder.trainer.validatefreq=1 \
encoder.trainer.niters=30 \
encoder.trainer.batch=16 \
encoder.sampler.name=pair \
encoder.trainer.gradacc=1
```

##### repbert-yang19-finetune-eval.sh
```
sbatch --export=ALL,fold=s1 repbert-yang19-finetune-eval-fold.sh
sbatch --export=ALL,fold=s2 repbert-yang19-finetune-eval-fold.sh
sbatch --export=ALL,fold=s3 repbert-yang19-finetune-eval-fold.sh
sbatch --export=ALL,fold=s4 repbert-yang19-finetune-eval-fold.sh
sbatch --export=ALL,fold=s5 repbert-yang19-finetune-eval-fold.sh
```

##### repbert-yang19-finetune-eval-fold.sh
```
#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /home/kjose/slurm-scripts/outputs/denserank-eval-yang19-finetune-all.log
#SBATCH -t 23:59:30

eval "$(conda shell.bash hook)"
conda activate capri
export JAVA_HOME=/home/kjose/jdk-11/
export CAPREOLUS_CACHE="/GW/NeuralIR/nobackup/kevin_cache"
export CAPREOLUS_RESULTS="/GW/NeuralIR/nobackup/kevin_results"
export PYTHONPATH=$PYTHONPATH:/home/kjose/capreolus
export PATH="/home/kjose/trec_eval:$PATH"
export CAPREOLUS_LOGGING=DEBUG
export CAPREOLUS_FAISS_LOG=/home/kjose/slurm-scripts/faiss_logs/faiss_denserank_eval_yang19_finetune_${fold}.log

echo $CUDA_VISIBLE_DEVICES

python /home/kjose/capreolus/capreolus/run.py denserank.evaluate with \
 numshards=400 \
 benchmark.name=robust04.yang19 \
 encoder.extractor.usecache=True \
 encoder.name=repbert \
 encoder.trainer.lr=0.000003 \
 fold=${fold} \
 encoder.trainer.validatefreq=1 \
 encoder.trainer.niters=30 \
 encoder.trainer.batch=16 \
 encoder.sampler.name=pair \
 encoder.trainer.gradacc=1
```
