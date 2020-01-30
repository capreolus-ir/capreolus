from collections import defaultdict

from capreolus.utils.loginit import get_logger
from tqdm import tqdm
import pytrec_eval
import numpy as np
import jnius_config
import os

from capreolus.collection import COLLECTIONS
from capreolus.utils.common import get_capreolus_base_dir

if not jnius_config.get_classpath() == "{0}/capreolus/anserini-0.7.0-fatjar.jar".format(get_capreolus_base_dir()):
    try:
        jnius_config.set_classpath("{0}/capreolus/anserini-0.7.0-fatjar.jar".format(get_capreolus_base_dir()))
    except:
        raise Exception("The classpath is: {0}".format(jnius_config.get_classpath()))


import functools
import torch

from capreolus.pipeline import Pipeline, cli_module_choice, modules
logger = get_logger(__name__)


def validate_datasources(data_sources):
    if data_sources is not None:
        if data_sources.get("qrels") is not None and not os.path.isfile(data_sources["qrels"]):
            raise FileNotFoundError(data_sources["qrels"])
        if data_sources.get("topics") is not None and not os.path.isfile(data_sources["topics"]):
            raise FileNotFoundError(data_sources["topics"])


def train_pipeline(pipeline_config, data_sources=None, early_stopping=False):
    pipeline = Pipeline(pipeline_config)
    # Ugly hack
    pipeline_config["earlystopping"] = early_stopping
    collection_name = pipeline_config["collection"]
    validate_datasources(data_sources)
    if data_sources is not None and data_sources.get("qrels") is not None:
        COLLECTIONS[collection_name].set_qrels(data_sources["qrels"])
    if data_sources is not None and data_sources.get("topics") is not None:
        COLLECTIONS[collection_name].set_topics(data_sources["topics"])
    if data_sources is not None and data_sources.get("documents") is not None:
        COLLECTIONS[collection_name].set_documents(data_sources["documents"])

    pipeline.ex.main(_train)

    run = pipeline.ex.run(config_updates=pipeline_config)
    return run.result


def _train(_config):
    pipeline_config = _config
    early_stopping = pipeline_config["earlystopping"]
    pipeline = Pipeline(pipeline_config)
    pipeline.initialize(pipeline_config)
    reranker = pipeline.reranker
    benchmark = pipeline.benchmark
    fold = benchmark.folds.get(pipeline.cfg["fold"], None)
    datagen = benchmark.training_tuples(fold["train_qids"])
    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_path = os.path.join(run_path, "weights")

    prepare_batch = functools.partial(_prepare_batch_with_strings, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    batches_per_epoch = pipeline_config["itersize"] // pipeline_config["batch"]
    batches_per_step = pipeline_config.get("gradacc", 1)

    optimizer = reranker.get_optimizer()
    best_accuracy = 0

    for niter in range(pipeline.cfg["niters"]):
        reranker.model.train()
        reranker.next_iteration()

        for bi, data in enumerate(datagen):
            data = prepare_batch(data)

            tag_scores = reranker.score(data)
            loss = pipeline.lossf(tag_scores[0], tag_scores[1], pipeline.cfg["batch"])
            loss.backward()

            if bi % batches_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                break

        if early_stopping:
            current_accuracy = max(evaluate_pipeline(pipeline))
            if current_accuracy > best_accuracy:
                logger.debug("Current accuracy: {0} is greater than best so far: {1}".format(current_accuracy, best_accuracy))
                best_accuracy = current_accuracy
                reranker.save(os.path.join(weight_path, "dev"))

    # TODO: Do early stopping to return the best instance of the reranker
    if early_stopping:
        reranker.load(os.path.join(weight_path, "dev"))

    return pipeline


def evaluate_pipeline(pipeline):
    pred_folds = {}
    pred_fold_sizes = {}
    benchmark = pipeline.benchmark
    fold = benchmark.folds.get(pipeline.cfg["fold"], None)
    reranker = pipeline.reranker

    # prepare generators
    if pipeline.cfg["predontrain"]:
        pred_fold_sizes[pipeline.cfg["fold"]] = sum(
            1 for qid in fold["train_qids"] for docid in benchmark.pred_pairs[qid])
        pred_folds[pipeline.cfg["fold"]] = (
            fold["train_qids"],
            predict_generator(pipeline.cfg, fold["train_qids"], benchmark),
        )
    for pred_fold, pred_qids in fold["predict"].items():
        pred_fold_sizes[pred_fold] = sum(1 for qid in pred_qids for docid in benchmark.pred_pairs[qid])
        pred_folds[pred_fold] = (pred_qids, predict_generator(pipeline.cfg, pred_qids, benchmark))


    prepare_batch = functools.partial(_prepare_batch_with_strings, device=pipeline.device)

    reranker.model.eval()
    ndcg_vals = []
    for pred_fold, (pred_qids, pred_gen) in pred_folds.items():
        pred_gen = iter(pred_gen)
        preds = predict(pipeline, pred_gen, reranker, prepare_batch)
        missing_qids = set(qid for qid in pred_qids if qid not in preds or len(preds[qid]) == 0)
        if len(missing_qids) > 0:
            raise RuntimeError(
                "predictions for some qids are missing, which may cause trec_eval's output to be incorrect\nmissing qids: %s"
                % missing_qids
            )

        test_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in pred_qids}
        fold_metrics = eval_preds(test_qrels, preds)

        if pred_fold == "dev":
            ndcg_vals.append(fold_metrics["ndcg_cut_20"])

    return ndcg_vals


def _prepare_batch_with_strings(batch, device, skip_strings=("qid", "posdocid", "negdocid")):

    def process(v):
        if v and isinstance(v[0], torch.Tensor):
            # Hack to make deeptilebars work. The deeptile extractor's posdoc output is a multi-dim tensor
            # We can't simply do np.array(multidim_tensor). Hence the if checks to treat it differently
            return torch.cat(v).to(device)
        else:
            return torch.tensor(np.array(v)).to(device)

    return {k: v if k in skip_strings else process(v) for k, v in batch.items()}


def predict_generator(p, qids, benchmark):
    qids = set(qids)
    pred_pairs = {qid: docids for qid, docids in benchmark.pred_pairs.items() if qid in qids}
    gen = benchmark.pred_tuples(pred_pairs)
    return gen


def predict(pipeline, gen, model, prepare_batch):
    preds = defaultdict(dict)
    with torch.autograd.no_grad():
        for data in tqdm(gen):
            qid_batch, docid_batch = data["qid"], data["posdocid"]
            data = prepare_batch(data)

            if pipeline.cfg["reranker"].startswith("Cedr"):
                scores = model.test(data)
            else:
                query, query_idf, doc = data["query"], data["query_idf"], data["posdoc"]
                scores = model.test(query, query_idf, doc, qids=qid_batch, posdoc_ids=docid_batch)
            scores = scores.view(-1).cpu().numpy()
            for qid, docid, score in zip(qid_batch, docid_batch, scores):
                # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                preds[qid][docid] = score.astype(np.float16).item()

    return preds


def eval_preds(test_qrels, target_preds):
    dev_eval = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut", "P", "map"})
    result = dev_eval.evaluate(target_preds)
    fold_metrics = defaultdict(list)

    for qid, metrics in result.items():
        for metric, val in metrics.items():
            fold_metrics[metric].append(val)

    return {key: np.mean(vals) for key, vals in fold_metrics.items()}