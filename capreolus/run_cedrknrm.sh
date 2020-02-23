CUDA_VISIBLE_DEVICES=1 nohup python train.py with \
  reranker=CEDRKNRM \
  collection=rob04_cedr \
  benchmark=robust04.title.cedr \
  niters=2 itersize=512\
  niters=50 vanillaiters=50 expid=cedrknrm_cmd \
  earlystopping=False esepoch=15 freezebert=False batch=2 gradacc=8 &
