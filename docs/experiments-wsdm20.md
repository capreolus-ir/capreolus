# Example: Reproducing Experiments on Robust04
## WSDM 2020

This page documents the commands used in the following article to evaluate NIR models on robust04:
> Andrew Yates, Siddhant Arora, Xinyu Zhang, Wei Yang, Kevin Martin Jose, and Jimmy Lin. 2020. Capreolus: A Toolkit for End-to-End Neural Ad Hoc Retrieval. In _Proceedings of the 13th International Conference on Web Search and Data Mining_ (_WSDM ’20_). 

These instructions assume you have installed and configured Capreolus [as described in the documentation](installation.md), including setting relevant environment variables like `CAPREOLUS_RESULTS`, `CAPREOLUS_CACHE`, and `CUDA_VISIBLE_DEVICES`.

### Commands
Each row from Table 1 in the above article requires several commands: five to perform training and inference using each of the five folds and one to compute metrics across folds. We list them for each reranker below.

If you are using zsh rather than bash, replace `$CFG` with `${=CFG}` in the lines below, and do the same for `$SHARED_CFG`. (The `$SHARED_CFG` variable is the same for every model below, but it is repeated in each section for convenience.)

#### DRMM
```
export CFG="reranker=DRMM histType=LCH gateType=IDF nodes=5 nbins=29"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### KNRM
```
export CFG="reranker=KNRM"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### ConvKNRM
```
export CFG="reranker=ConvKNRM filters=300"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### PACRR
```
export CFG="reranker=PACRR nfilters=32 kmax=2"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### POSITDRMM
```
export CFG="reranker=POSITDRMM lr=0.001"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### HINT
```
export CFG="reranker=HINT LSTMdim=6 kmax=10 batch=128 lr=5e-3"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### DeepTileBars
```
export CFG="reranker=DeepTileBar"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### DUET
```
export CFG="reranker=DUET"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### DSSM
```
export CFG="reranker=DSSM datamode=trigram"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

#### CDSSM
```
export CFG="reranker=CDSSM nfilter=1 nhiddens=30 nkernel=30 windowsize=4"
export SHARED_CFG="benchmark=robust04.title.wsdm20demo expid=reproduce.wsdm20demo niters=50 keepstops=False"
for FOLDIDX in s1 s2 s3 s4 s5; do
  python train.py with $CFG $SHARED_CFG fold=$FOLDIDX
done
python train.py evaluate with $CFG $SHARED_CFG fold=$FOLDIDX
```

### Non-determinism
Capreolus controls the random seeds used by Python and the libraries it uses (numpy, PyTorch, etc). Some CUDA operations are still non-deterministic, however, [as described in PyTorch's documentation](https://pytorch.org/docs/stable/notes/randomness.html):
> One class of such CUDA functions are atomic operations, in particular  `atomicAdd`, where the order of parallel additions to the same value is undetermined and, for floating-point variables, a source of variance in the result. ... There currently is no simple way of avoiding non-determinism in these functions.

As a result of this, your results are likely to vary slightly depending on the GPU and environment used. This appears to be correlated with model complexity. In our observations, smaller models like DRMM, KNRM, and PACRR vary 1% or less across systems with different hardware (e.g., GPU and CPU generations). We note that such variation should not be statistically significant.
