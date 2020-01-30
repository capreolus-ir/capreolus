import functools
import os
import resource
import shutil
import subprocess
import sys
import json
import time

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pytrec_eval
import torch

from tqdm import tqdm
from scipy.stats import ttest_rel

curr_file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_file_dir)

from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus.utils.loginit import get_logger
from capreolus.pipeline import Pipeline, cli_module_choice, modules
from capreolus.searcher import Searcher

logger = get_logger(__name__)  # pylint: disable=invalid-name
plt.switch_backend("agg")

pipeline = Pipeline({module: cli_module_choice(sys.argv, module) for module in modules})
pipeline.ex.logger = logger


@pipeline.ex.main
def train(_config, _run):
    pipeline.initialize(_config)
    reranker = pipeline.reranker
    benchmark = pipeline.benchmark
    logger.debug("initialized pipeline with results path: %s", pipeline.reranker_path)
    post_pipeline_init_time = time.time()
    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    logger.info("initialized pipeline with results path: %s", run_path)
    post_pipeline_init_time = time.time()
    info_path = os.path.join(run_path, "info")
    os.makedirs(info_path, exist_ok=True)
    weight_path = os.path.join(run_path, "weights")
    os.makedirs(weight_path, exist_ok=True)
    predict_path = os.path.join(run_path, "predict")

    if pipeline.cfg["dataparallel"] == "gpu":
        reranker.to(pipeline.device)
        if torch.cuda.device_count() > 1:
            reranker.model = torch.nn.DataParallel(reranker.model)
        else:
            logger.warning("ignoring dataparallel=gpu because only %s CUDA device(s) can be found", torch.cuda.device_count())
    elif pipeline.cfg["dataparallel"] == "tpu":
        pass
        # import torch_xla_py.xla_model as xm
        # import torch_xla_py.data_parallel as dp

        # xla_devices = xm.get_xla_supported_devices()
        # xla_dp = dp.DataParallel(model.model, device_ids=xla_devices)
    else:
        reranker.to(pipeline.device)

    optimizer = reranker.get_optimizer()

    fold = benchmark.folds.get(pipeline.cfg["fold"], None)

    if fold is None:
        raise RuntimeError(
            f"invalid fold for benchmark {pipeline.cfg['benchmark']}: {pipeline.cfg['fold']}\n"
            + f"valid folds: {benchmark.folds.keys()}"
        )

    for module, ks in sorted(pipeline.module_to_parameters.items()):
        logger.debug(module + ": config options: %s", {k: pipeline.cfg[k] for k in ks})

    prepare_batch = functools.partial(_prepare_batch_with_strings, device=pipeline.device)
    datagen = benchmark.training_tuples(fold["train_qids"])

    # folds to predict on
    pred_folds = {}
    pred_fold_sizes = {}
    # prepare generators
    if pipeline.cfg["predontrain"]:
        pred_fold_sizes[pipeline.cfg["fold"]] = sum(1 for qid in fold["train_qids"] for docid in benchmark.pred_pairs[qid])
        pred_folds[pipeline.cfg["fold"]] = (fold["train_qids"], predict_generator(pipeline.cfg, fold["train_qids"], benchmark))
    for pred_fold, pred_qids in fold["predict"].items():
        pred_fold_sizes[pred_fold] = sum(1 for qid in pred_qids for docid in benchmark.pred_pairs[qid])
        pred_folds[pred_fold] = (pred_qids, predict_generator(pipeline.cfg, pred_qids, benchmark))

    metrics = {}
    initial_iter = 0
    history = []
    batches_since_update = 0
    dev_ndcg_max = -1
    batches_per_epoch = pipeline.cfg["itersize"] // pipeline.cfg["batch"]
    batches_per_step = pipeline.cfg.get("gradacc", 1)
    pbar_loop = tqdm(desc="loop", total=pipeline.cfg["niters"], initial=initial_iter, position=0, leave=True, smoothing=0.0)
    pbar_train = tqdm(
        desc="training",
        total=pipeline.cfg["niters"] * pipeline.cfg["itersize"],
        initial=initial_iter * pipeline.cfg["itersize"],
        unit_scale=True,
        position=1,
        leave=True,
    )
    dev_best_info = ""
    logger.info("It took {0} seconds to reach training loop after pipeline init".format(post_pipeline_init_time - time.time()))
    pbar_info = tqdm(position=2, leave=True, bar_format="{desc}")
    for niter in range(initial_iter, pipeline.cfg["niters"]):
        reranker.model.train()
        reranker.next_iteration()
        iter_loss = []

        for bi, data in enumerate(datagen):
            data = prepare_batch(data)
            pbar_train.update(pipeline.cfg["batch"])

            tag_scores = reranker.score(data)
            loss = pipeline.lossf(tag_scores[0], tag_scores[1], pipeline.cfg["batch"])
            # loss /= batches_per_step
            iter_loss.append(loss.item())
            loss.backward()
            batches_since_update += 1
            if batches_since_update >= batches_per_step:
                batches_since_update = 0
                optimizer.step()
                optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                break

        avg_loss = np.mean(iter_loss)
        pbar_info.set_description_str(f"loss: {avg_loss:0.5f}\t{dev_best_info}{'':40s}")
        # logger.info("epoch = %d loss = %f", niter, avg_loss)

        # make predictions
        reranker.model.eval()
        for pred_fold, (pred_qids, pred_gen) in pred_folds.items():
            pbar_info.set_description_str(
                f"loss: {avg_loss:0.5f}\t{dev_best_info}\t[predicting {pred_fold_sizes[pred_fold]} pairs]"
            )
            pred_gen = iter(pred_gen)
            predfn = os.path.join(predict_path, pred_fold, str(niter))
            os.makedirs(os.path.dirname(predfn), exist_ok=True)

            preds = predict_and_save_to_file(pred_gen, reranker, predfn, prepare_batch)
            missing_qids = set(qid for qid in pred_qids if qid not in preds or len(preds[qid]) == 0)
            if len(missing_qids) > 0:
                raise RuntimeError(
                    "predictions for some qids are missing, which may cause trec_eval's output to be incorrect\nmissing qids: %s"
                    % missing_qids
                )

            test_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in pred_qids}
            fold_metrics = eval_preds_niter(test_qrels, preds, niter)

            if pred_fold == "dev" and fold_metrics["ndcg_cut_20"][1] >= dev_ndcg_max:
                dev_ndcg_max = fold_metrics["ndcg_cut_20"][1]
                # logger.info("saving best dev model with dev ndcg@20: %0.3f", dev_ndcg_max)
                dev_best_info = "dev best: %0.3f on iter %s" % (dev_ndcg_max, niter)
                reranker.save(os.path.join(weight_path, "dev"))

            for metric, (x, y) in fold_metrics.items():
                metrics.setdefault(pred_fold, {}).setdefault(metric, []).append((x, y))

        pbar_loop.update(1)
        pbar_info.set_description_str(f"loss: {avg_loss:0.5f}\t{dev_best_info}{'':40s}")
        history.append((niter, avg_loss))

    pbar_train.close()
    pbar_loop.close()
    pbar_info.close()

    logger.info(dev_best_info)
    with open(os.path.join(info_path, "loss.txt"), "wt") as outf:
        for niter, loss in history:
            print("%s\t%s" % (niter, loss), file=outf)
            # _run.log_scalar("train.loss", loss, niter)

    plot_loss_curve(history, os.path.join(info_path, "loss.pdf"))
    plot_metrics(metrics, predict_path)

    with open("{0}/config.json".format(run_path), "w") as fp:
        json.dump(_config, fp)


def eval_preds_niter(test_qrels, target_preds, niter):
    dev_eval = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut", "P", "map"})
    result = dev_eval.evaluate(target_preds)
    fold_metrics = defaultdict(list)

    for qid, metrics in result.items():
        for metric, val in metrics.items():
            fold_metrics[metric].append(val)

    return {key: (niter, np.mean(vals)) for key, vals in fold_metrics.items()}


def predict_generator(p, qids, benchmark):
    qids = set(qids)
    pred_pairs = {qid: docids for qid, docids in benchmark.pred_pairs.items() if qid in qids}
    gen = benchmark.pred_tuples(pred_pairs)
    return gen


# TODO need to randomly switch doc order in case of score ties,
def predict_and_save_to_file(gen, model, outfn, prepare_batch):
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

    # logger.info("predicted scores for %s pairs", sum(1 for qid in preds for docid in preds[qid]))

    # logger.info("writing predictions file: %s", outfn)
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    Searcher.write_trec_run(preds, outfn)

    return preds


def plot_loss_curve(history, outfn):
    epochs, losses = zip(*history)
    best_epoch = epochs[np.argmin(losses)]
    fig = plt.figure()
    plt.plot(epochs, losses, "k-.")
    plt.ylabel("Training Loss")
    plt.tick_params("y")
    plt.xlabel("Iteration")
    plt.title("min loss: %d %.3f" % (best_epoch, losses[best_epoch]))
    fig.savefig(outfn, format="pdf")
    plt.close()


def plot_metrics(metrics, outdir, show={"map", "P_20", "ndcg_cut_20"}):
    for fold in metrics:
        outfn = os.path.join(outdir, f"{fold}.pdf")
        title = "maxs: "
        for metric, xys in metrics[fold].items():
            if metric not in show:
                continue
            plt.plot(*zip(*xys), label=metric)
            max_iter, max_metric = max(xys, key=lambda x: x[1])
            title += f"{metric} {max_metric:0.3f} ({max_iter}) "

        plt.ylabel("Metric")
        plt.tick_params("y")
        plt.xlabel("Iteration")
        plt.title(title)
        plt.legend()
        plt.savefig(outfn, format="pdf")
        plt.close()


@pipeline.ex.command
def evaluate(_config):
    from capreolus.searcher import Searcher
    import pytrec_eval

    pipeline.initialize(_config)
    logger.debug("initialized pipeline with results path: %s", pipeline.reranker_path)

    benchmark = pipeline.benchmark
    benchmark.build()  # TODO move this to pipeline.initialize?

    test_metrics = {}
    searcher_test_metrics = {}
    interpolated_test_metrics = {}
    for foldname, fold in sorted(benchmark.folds.items()):
        if not (len(fold["predict"]) == 2 and "dev" in fold["predict"] and "test" in fold["predict"]):
            raise RuntimeError("this evaluation command is only supported for benchmarks with 'dev' and 'test' folds")

        logger.debug("evaluating fold: %s", foldname)
        predict_path = os.path.join(pipeline.reranker_path, foldname, "predict")

        dev_qids = set(fold["predict"]["dev"])
        dev_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in dev_qids}
        dev_eval = pytrec_eval.RelevanceEvaluator(dev_qrels, {"ndcg_cut", "P", "map"})

        best_metric, best_iter, dev_run = -np.inf, None, None
        target_metric = "ndcg_cut_20"
        # target_metric = "map"
        devpath = os.path.join(predict_path, "dev")
        for iterfn in os.listdir(devpath):
            run = Searcher.load_trec_run(os.path.join(devpath, iterfn))
            this_metric = np.mean([q[target_metric] for q in dev_eval.evaluate(run).values()])
            if this_metric > best_metric:
                best_metric = this_metric
                best_iter = iterfn
                dev_run = run
        logger.debug("best dev %s=%0.3f was on iteration #%s", target_metric, best_metric, best_iter)

        test_run = Searcher.load_trec_run(os.path.join(predict_path, "test", best_iter))
        test_qids = set(fold["predict"]["test"])
        test_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in test_qids}
        test_eval = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut", "P", "map"})
        for qid, metrics in test_eval.evaluate(test_run).items():
            assert qid in test_qids
            for metric, value in metrics.items():
                test_metrics.setdefault(metric, {})
                assert qid not in test_metrics[metric], "fold testqid overlap"
                test_metrics[metric][qid] = value

        # compute metrics for the run being reranked
        for qid, metrics in test_eval.evaluate(benchmark.reranking_runs[foldname]).items():
            assert qid in test_qids
            for metric, value in metrics.items():
                searcher_test_metrics.setdefault(metric, {})
                assert qid not in searcher_test_metrics[metric], "fold testqid overlap"
                searcher_test_metrics[metric][qid] = value

        # choose an alpha for interpolation using the dev_qids,
        # then create a run by interpolating the searcher and reranker scores
        searcher_dev = {qid: docscores for qid, docscores in benchmark.reranking_runs[foldname].items() if qid in dev_qids}
        searcher_test = {qid: docscores for qid, docscores in benchmark.reranking_runs[foldname].items() if qid in test_qids}
        alpha, interpolated_test_run, _ = Searcher.crossvalidated_interpolation(
            dev={"reranker": dev_run, "searcher": searcher_dev, "qrels": dev_qrels},
            test={"reranker": test_run, "searcher": searcher_test, "qrels": test_qrels},
            metric=target_metric,
        )

        # output files for Anserini interpolation script
        Searcher.write_trec_run(dev_run, f"runs.reranker.{foldname}.dev")
        Searcher.write_trec_run(test_run, f"runs.reranker.{foldname}.test")
        Searcher.write_trec_run(searcher_dev, f"runs.searcher.{foldname}.dev")
        Searcher.write_trec_run(searcher_test, f"runs.searcher.{foldname}.test")

        logger.debug(f"interpolation alpha={alpha}")
        for qid, metrics in test_eval.evaluate(interpolated_test_run).items():
            assert qid in test_qids
            for metric, value in metrics.items():
                interpolated_test_metrics.setdefault(metric, {})
                assert qid not in interpolated_test_metrics[metric], "fold testqid overlap"
                interpolated_test_metrics[metric][qid] = value

    logger.info(f"optimized for {target_metric}")
    logger.info(f"results on {len(test_metrics[metric])} aggregated test qids")
    for metric in ["map", "P_20", "ndcg_cut_20"]:
        assert len(test_metrics[metric]) == len(searcher_test_metrics[metric])
        assert len(test_metrics[metric]) == len(interpolated_test_metrics[metric])

        searcher_avg = np.mean([*searcher_test_metrics[metric].values()])
        logger.info(f"[searcher] avg {metric}: {searcher_avg:0.3f}")

        sigtest_qids = sorted(test_metrics[metric].keys())
        sigtest = ttest_rel(
            [searcher_test_metrics[metric][qid] for qid in sigtest_qids], [test_metrics[metric][qid] for qid in sigtest_qids]
        )

        avg = np.mean([*test_metrics[metric].values()])
        logger.info(f"[reranker] avg {metric}: {avg:0.3f}\tp={sigtest.pvalue:0.3f} (vs. searcher)")

        interpolated_avg = np.mean([*interpolated_test_metrics[metric].values()])
        logger.info(f"[interpolated] avg {metric}: {interpolated_avg:0.3f}")

    with open(os.path.join(predict_path, "results.json"), "wt") as outf:
        json.dump((test_metrics, searcher_test_metrics, interpolated_test_metrics), outf)


@pipeline.ex.command
def interpolate(_config):
    from capreolus.searcher import Searcher
    import pytrec_eval

    pipeline.initialize(_config)
    logger.info("initialized pipeline with results path: %s", pipeline.reranker_path)

    benchmark = pipeline.benchmark
    benchmark.build()  # TODO move this to pipeline.initialize?

    test_metrics = {}
    for foldname, fold in sorted(benchmark.folds.items()):
        if not (len(fold["predict"]) == 2 and "dev" in fold["predict"] and "test" in fold["predict"]):
            raise RuntimeError("this evaluation command is only supported for benchmarks with 'dev' and 'test' folds")

        logger.debug("evaluating fold: %s", foldname)
        predict_path = os.path.join(pipeline.reranker_path, foldname, "predict")

        dev_qids = set(fold["predict"]["dev"])
        dev_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in dev_qids}
        dev_eval = pytrec_eval.RelevanceEvaluator(dev_qrels, {"ndcg_cut", "P", "map"})

        test_qids = set(fold["predict"]["test"])
        test_qrels = {qid: labels for qid, labels in pipeline.collection.qrels.items() if qid in test_qids}
        searcher_dev = {qid: docscores for qid, docscores in benchmark.reranking_runs[foldname].items() if qid in dev_qids}
        searcher_test = {qid: docscores for qid, docscores in benchmark.reranking_runs[foldname].items() if qid in test_qids}

        best_metric, best_iter, dev_run = -np.inf, None, None
        target_metric = "ndcg_cut_20"
        # target_metric = "map"
        devpath = os.path.join(predict_path, "dev")
        for iterfn in os.listdir(devpath):
            dev_run = Searcher.load_trec_run(os.path.join(devpath, iterfn))
            test_run = Searcher.load_trec_run(os.path.join(predict_path, "test", iterfn))
            alpha, interpolated_test_run, interpolated_dev_run = Searcher.crossvalidated_interpolation(
                dev={"reranker": dev_run, "searcher": searcher_dev, "qrels": dev_qrels},
                test={"reranker": test_run, "searcher": searcher_test, "qrels": test_qrels},
                metric=target_metric,
            )

            this_metric = np.mean([q[target_metric] for q in dev_eval.evaluate(interpolated_dev_run).values()])
            if this_metric > best_metric:
                best_metric = this_metric
                best_iter = iterfn
                use_run = interpolated_test_run
                print(foldname, iterfn, best_metric, alpha)
        logger.debug("best dev %s was on iteration #%s", target_metric, best_iter)

        # test_run = Searcher.load_trec_run(os.path.join(predict_path, "test", best_iter))
        test_run = use_run
        test_eval = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut", "P", "map"})
        for qid, metrics in test_eval.evaluate(test_run).items():
            assert qid in test_qids
            for metric, value in metrics.items():
                test_metrics.setdefault(metric, {})
                assert qid not in test_metrics[metric], "fold testqid overlap"
                test_metrics[metric][qid] = value

        # output files for Anserini interpolation script
        Searcher.write_trec_run(
            Searcher.load_trec_run(os.path.join(predict_path, "dev", best_iter)), f"runs.rerankerIES.{foldname}.dev"
        )
        Searcher.write_trec_run(
            Searcher.load_trec_run(os.path.join(predict_path, "test", best_iter)), f"runs.rerankerIES.{foldname}.test"
        )

    logger.info(f"optimized for {target_metric}")
    logger.info(f"results on {len(test_metrics[metric])} aggregated test qids")
    for metric in ["ndcg_cut_20", "map", "P_5", "P_20"]:
        interpolated_avg = np.mean([*test_metrics[metric].values()])
        logger.info(f"[interpolated] avg {metric}: {interpolated_avg:0.3f}")


@pipeline.ex.command
def choices():
    from capreolus.reranker.reranker import Reranker
    from capreolus.collection import COLLECTIONS
    from capreolus.benchmark import Benchmark
    from capreolus.index import Index
    from capreolus.searcher import Searcher

    module_loaders = {
        "collection": COLLECTIONS,
        "index": Index.ALL,
        "searcher": Searcher.ALL,
        "benchmark": Benchmark.ALL,
        "reranker": Reranker.ALL,
    }

    print(f"{'<module>': <20} <name>")
    for module in sorted(modules):
        for value in sorted(module_loaders[module]):
            print(f"{module: <20} {value}")


def _prepare_batch_with_strings(batch, device, skip_strings=("qid", "posdocid", "negdocid")):
    def process(v):
        if v and isinstance(v[0], torch.Tensor):
            # Hack to make deeptilebars work. The deeptile extractor's posdoc output is a multi-dim tensor
            # We can't simply do np.array(multidim_tensor). Hence the if checks to treat it differently
            return torch.cat(v).to(device, non_blocking=True)
        else:
            return torch.tensor(np.array(v)).to(device, non_blocking=True)

    return {k: v if k in skip_strings else process(v) for k, v in batch.items()}


if __name__ == "__main__":
    pipeline.ex.run_commandline()

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # in KB
    logger.debug("maxrss: %0.2f GB (does not include any forked processes)", maxrss)

    try:
        logger.debug(
            "torch.cuda.max_memory_allocated on device %s: %0.2f GB",
            pipeline.device,
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
        )
        logger.debug(
            "torch.cuda.max_memory_reserved on device %s: %0.2f GB",
            pipeline.device,
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
        )
    except:  # AssertionError as ae:
        # logger.debug("unable to retrieve CUDA memory usage due to exception: %s", ae)
        pass
