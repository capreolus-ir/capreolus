from collections import defaultdict

import torch
import os
import json
import numpy as np
import time
from mock import patch

from django.http import JsonResponse
from django.shortcuts import render
from django.utils.datastructures import MultiValueDict
from django.views import View
from django.views.generic import TemplateView

from capreolus.collection import Collection
from capreolus.demo_app.constants import WEIGHTS_DIR_NAME, WEIGHTS_FILE_NAME, NUM_RESULTS_TO_SHOW
from capreolus.demo_app.forms import QueryForm, BM25Form
from capreolus.demo_app.utils import search_files_or_folders_in_directory, copy_keys_to_each_other
from capreolus.extractor.embedding import EmbeddingHolder
from capreolus.index import Index
from capreolus.reranker.reranker import Reranker
from capreolus.pipeline import Pipeline
from capreolus.tokenizer import Tokenizer
from capreolus.utils.common import padlist, get_default_cache_dir, get_default_results_dir
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class ConfigsView(TemplateView):
    """
    Returns a dict of configurations based on available experiment results
    """

    template_name = "demo.html"

    def get(self, request, *args, **kwargs):
        available_configs = self.get_config_from_results()
        available_indices = self.get_available_indices()
        context = {"configs": available_configs, "available_indices": available_indices}

        return render(request, self.template_name, context)

    @staticmethod
    def get_available_indices():
        cache_path = os.environ.get("CAPREOLUS_CACHE", get_default_cache_dir())
        index_dirs = search_files_or_folders_in_directory(cache_path, "index")
        index_dirs_with_done = [
            index_dir for index_dir in index_dirs if len(search_files_or_folders_in_directory(index_dir, "done"))
        ]

        return index_dirs_with_done

    @staticmethod
    def get_config_from_results():
        # TODO: Using 'results' if there's no env varialbe - this logic is duplicated. Move to a single place?
        results_folder = os.environ.get("CAPREOLUS_RESULTS", get_default_results_dir())
        config_files = search_files_or_folders_in_directory(results_folder, "config.json")
        configs = []

        for file in config_files:
            with open(file) as json_file:
                data = json.load(json_file)
                configs.append(data)

        return configs


class ExperimentView(TemplateView):
    """
        Displays a query box, along with the chosen experiment configs
    """

    template_name = "experiment.html"

    def get(self, request, *args, **kwargs):
        context = {"config": sorted(request.GET.items())}
        return render(request, self.template_name, context)


class QuerySuggestionView(View):
    """
    Used in auto-completing query suggestions in the demo page
    This works only if the index that the user chose was created from a collection
    """

    def get(self, request, *args, **kwargs):
        query = request.GET.dict()["q"]
        target_index = request.GET.dict()["target_index"]
        collection = Collection.get_collection_from_index_path(target_index)

        if collection is None:
            return JsonResponse([], safe=False)
        else:
            suggestions = collection.get_query_suggestions(query)
            return JsonResponse(suggestions, safe=False)


class QueryView(TemplateView):
    def get(self, request, *args, **kwargs):
        model_name = request.GET.dict()["reranker"]
        if model_name == "BM25":
            return BM25View().get(request, *args, **kwargs)
        else:
            return NeuralQueryView().get(request, *args, **kwargs)


class NeuralQueryView(TemplateView):
    """
        Perform a query against the chosen pipeline and display the results
    """

    template_name = "experiment.html"

    def get(self, request, *args, should_render=True, **kwargs):
        # The expected query params differ based on the NIR model. We use django forms to help us do the validation
        formcls = QueryForm.FORM_CLASSES[request.GET.dict()["reranker"]]
        form = formcls(request.GET.dict())

        if form.is_valid():
            # TODO: Keep a list of allowed/known experiment config and use that to extract config from request.GET
            # request.GET might contain key value pairs that won't be recognized by sacred
            query_string = form.cleaned_data.pop("query")
            target_index = form.cleaned_data.pop("target_index")
            config = form.cleaned_data

            results = self.get_most_relevant_doc_based_on_config(config, query_string, target_index)
            # Hack. Re-adding target_index because the html page would need it for subsequent queries
            config["target_index"] = target_index
            context = {
                "config": sorted(config.items()),
                "query": query_string,
                "has_result": True,
                "results": results,
                "target_index": target_index,
            }
            if should_render:
                return render(request, self.template_name, context)
            else:
                return context
        else:
            raise Exception("Malformed query: {0}".format(form.errors))

    # TODO: Move this to the Pipeline class maybe?
    @staticmethod
    def get_weight_file(pipeline):
        """
        Given a pipeline, get the path to a weight file.
        If there are multiple weight files, returns the path to the first one
        Here weight file = a file created from torch.model.save()
        """
        run_path = os.path.join(pipeline.model_path, pipeline.cfg["fold"])
        weight_dir = os.path.join(run_path, WEIGHTS_DIR_NAME)
        weight_file = search_files_or_folders_in_directory(weight_dir, WEIGHTS_FILE_NAME)
        return weight_file[0]

    @staticmethod
    def get_tokenizer(pipeline, config, tokenizer_name):
        tokenizer_class = Tokenizer.ALL[tokenizer_name]
        tokenizer = tokenizer_class(pipeline, keepstops=config.get("keepstops", False), use_cache=False)
        tokenizer.create()
        return tokenizer

    @staticmethod
    def get_tokens_from_docs_and_query(tokenizer, docs, query_string):
        """
        Returns one big list of tokens, based on the terms that appear in supplied documents and the query string
        """
        all_tokens = [tokenizer.tokenize(doc) for doc in docs]
        # Just flattening
        all_tokens = [token for sublist in all_tokens for token in sublist]
        all_tokens.extend(tokenizer.tokenize(query_string))
        return all_tokens

    @staticmethod
    def create_tensor_from_docs(docs, tokenizer, embedding_holder, maxdoclen):
        """
        Returns a stacked tensor, with each element being a document's tensor representation
        """
        doc_features = []
        for doc in docs:
            doc_tokens = tokenizer.tokenize(doc)
            # Doc features should be indices into the embedding layer. This is an optimization enforced by pytorch
            doc_token_indices = embedding_holder.get_index_array_from_tokens(doc_tokens, maxdoclen)
            doc_features.append(torch.from_numpy(doc_token_indices))

        return torch.stack(doc_features)

    @staticmethod
    def create_tensor_from_query_string(query_string, index, tokenizer, embedding_holder, doc_features_len, maxqlen):
        """
        Creates a tensor from the query string, and stacks it repeatedly (i.e duplicate) so as to match the dimensions
        of the document features tensor
        :param query_string
        :param tokenizer - A tokenizer instance
        :param embedding_holder - An instance of EmbeddingHolder
        :param doc_features_len - The length of the document features tensor. This dictates the size of the resulting
        query features tensor created by this method
        :param maxqlen - Maximum query length
        """
        query_toks = tokenizer.tokenize(query_string)
        query_feature = torch.from_numpy(embedding_holder.get_index_array_from_tokens(query_toks, maxqlen))
        query_features = torch.stack([query_feature for i in range(0, doc_features_len)])

        query_idfs = []
        for i in range(0, doc_features_len):
            query_idfs.append(torch.from_numpy(np.array(padlist([index.getidf(tok) for tok in query_toks], maxqlen))))
        query_idfs = torch.stack(query_idfs)

        return query_features, query_idfs

    @staticmethod
    def add_model_required_params_to_config(config, embedding_holder):
        # So that MODEL_BASE.validate_params() would pass
        # We are adding params required by all the NIR model classes. The alternative is to do if condition checks
        # for the model class and add only the relevant params. But this is cleaner, and we are not saving anything
        # with the if checks
        config["pad_token"] = 0
        config["stoi"] = embedding_holder.get_stoi()
        config["itos"] = embedding_holder.get_itos()
        config["nvocab"] = embedding_holder.get_nvocab()

        return config

    @staticmethod
    def construct_result_dicts(doc_ids, docs, relevances):
        result_dicts = []

        for i in range(len(doc_ids)):
            result_dicts.append({"doc_id": doc_ids[i], "doc": docs[i], "relevance": relevances[i]})
        return result_dicts

    @staticmethod
    def get_most_relevant_doc_based_on_config(config, query_string, target_index):
        """
        1. Instantiate various classes based on config
        2. Get the most relevant doc
        """
        # We still need to init a pipeline because it pre-processes some config params, and we rely on that to
        # construct paths e.t.c.
        config = config.copy()  # because we end up modifying config
        pipeline = Pipeline(config)
        pipeline.initialize(config)
        path_dict = pipeline.get_paths(config)
        index_path = target_index
        index_class = Index.get_index_from_index_path(index_path)
        index = index_class(pipeline.collection, index_path, None)  # TODO: Pass a proper index_key
        model_class = Reranker.ALL[config["reranker"]]
        tokenizer = NeuralQueryView.get_tokenizer(pipeline, config, index_class.name)
        embedding_holder = EmbeddingHolder.get_instance(config.get("embeddings", "glove6b"))
        trained_weight_path = path_dict["trained_weight_path"]
        config = NeuralQueryView.add_model_required_params_to_config(config, embedding_holder)

        return NeuralQueryView.do_query(
            config,
            query_string,
            pipeline,
            index,
            tokenizer,
            embedding_holder,
            model_class,
            trained_weight_path=trained_weight_path,
        )

    @staticmethod
    def do_query(config, query_string, pipeline, index, tokenizer, embedding_holder, model_class, trained_weight_path=None):
        """
        1. Do a bm25 search to get top 100 results. This is an optimization technique - we don't want to feed the entire
        dataset to the NIR model's forward pass
        2. Based on the documents retrieved above, create an embedding layer to be used in the NIR model
        3. Instantiate an NIR model, load the trained weights, and do a forward pass with the 1000 docs
        4. Return the document with the highest score
        """
        # 1. Do bm25 search and tokenize the results
        # TODO: Handle the case where bm25 search returns no results
        api_start = time.time()
        doc_ids, docs = BM25View.do_query(query_string, index)

        all_tokens = NeuralQueryView.get_tokens_from_docs_and_query(tokenizer, docs, query_string)

        # 2. Form an embedding layer and doc and query features
        embeddings = torch.from_numpy(embedding_holder.create_indexed_embedding_layer_from_tokens(all_tokens)).to(pipeline.device)
        doc_features = NeuralQueryView.create_tensor_from_docs(docs, tokenizer, embedding_holder, config["maxdoclen"]).to(
            pipeline.device
        )
        query_features, query_idf_features = NeuralQueryView.create_tensor_from_query_string(
            query_string, index, tokenizer, embedding_holder, len(doc_features), config["maxqlen"]
        )
        query_features = query_features.to(pipeline.device)
        query_idf_features = query_idf_features.to(pipeline.device)

        # 3. Do a forward pass of the NIR model class, and get the max scoring document
        # TODO: Remove the dependence of NIR reranker on pipeline. Pass the configs explicitly
        model_instance = model_class(embeddings, None, config)
        model_instance.build()
        model_instance.to(pipeline.device)
        # model_instance = model_class.alternate_init(pipeline, embeddings, config).to(pipeline.device)
        if trained_weight_path is not None:
            model_instance.load(trained_weight_path)
        scores = model_instance.test(query_features, query_idf_features, doc_features)
        max_scoring_doc_ids = [doc_ids[i] for i in reversed(torch.argsort(scores))]
        max_scoring_doc_ids = max_scoring_doc_ids[:NUM_RESULTS_TO_SHOW]
        # TODO: Get this in a single go, instead of calling `getdoc()` repeatedly
        docs = [index.getdoc(doc_id) for doc_id in max_scoring_doc_ids]
        # Trimming to first 250 chars. TODO: Make this configurable
        docs = [doc[:250] for doc in docs]

        relevance_fetch_start = time.time()
        collection = Collection.get_collection_from_index_path(index.index_path)
        relevances = collection.get_relevance(query_string, max_scoring_doc_ids) if collection is not None else [0] * len(doc_ids)
        result_dicts = NeuralQueryView.construct_result_dicts(max_scoring_doc_ids, docs, relevances)
        relevance_fetch_stop = time.time()

        api_stop = time.time()
        logger.debug("Took {0} seconds to get the most relevant doc".format(api_stop - api_start))
        logger.debug(
            "Determining the relevance of the fetched document took {0} seconds".format(
                relevance_fetch_stop - relevance_fetch_start
            )
        )
        return result_dicts


class DocumentView(TemplateView):
    """
        Displays a single document
    """

    template_name = "document.html"

    def get(self, request, *args, **kwargs):
        index_class = Index.get_index_from_index_path(request.GET["target_index"])
        index = index_class(request.GET["target_index"])
        doc_id = request.GET["doc_id"]
        context = {"doc": index.getdoc(doc_id)}

        return render(request, self.template_name, context)


class QueryDictParserMixin:
    @staticmethod
    def get_two_configs_from_query_dict(query_dict):
        """
        Parses the django query dict and separates it into two different experiment configs.
        eg: "/api/?model=KNRM&model=DRMM&blah..." should be parsed so that "model=KNRM" goes to one config while
        "model=DRMM" goes to the other
        """
        assert isinstance(query_dict, MultiValueDict)

        config_1, config_2 = dict(), dict()

        for key, values in query_dict.lists():
            if len(values) == 2:
                config_1[key] = values[0]
                config_2[key] = values[1]
            elif len(values) == 1:
                config_1[key] = values[0]
                config_2[key] = values[0]
            else:
                # TODO: This is a 500 error. Replace this with a 400 bad request error
                raise Exception("Bad request")

        return config_1, config_2


class CompareExperimentsView(TemplateView, QueryDictParserMixin):
    """
        Displays a query box, along with configs of 2 different experiments
    """

    def get(self, request, *args, **kwargs):
        config_1, config_2 = self.get_two_configs_from_query_dict(request.GET)
        zipped_configs = list(zip(sorted(config_1.items()), sorted(config_2.items())))
        context = {"configs": zipped_configs, "config_1": sorted(config_1.items()), "config_2": sorted(config_2.items())}

        return render(request, "experiment_comparison.html", context)


class CompareQueryView(TemplateView, QueryDictParserMixin):
    def get(self, request, *args, **kwargs):
        config_1, config_2 = self.get_two_configs_from_query_dict(request.GET)
        if "BM25" in [config_1["reranker"], config_2["reranker"]]:
            return CompareBM25AndNeuralQueryView().get(request, *args, **kwargs)
        else:
            return CompareNeuralQueryView().get(request, *args, **kwargs)


class CompareNeuralQueryView(TemplateView, QueryDictParserMixin):
    """
    Perform a query against two different NIR reranker and display the results for comparison
    """

    template_name = "experiment_comparison.html"

    def get(self, request, *args, **kwargs):
        config_1, config_2 = self.get_two_configs_from_query_dict(request.GET)
        formcls_1 = QueryForm.FORM_CLASSES[config_1["reranker"]]
        form_1 = formcls_1(config_1)
        formcls_2 = QueryForm.FORM_CLASSES[config_2["reranker"]]
        form_2 = formcls_2(config_2)

        if form_1.is_valid() and form_2.is_valid():
            query_string = form_1.cleaned_data.pop("query")
            form_2.cleaned_data.pop("query")
            target_index = form_1.cleaned_data.pop("target_index")
            form_2.cleaned_data.pop("target_index")

            config_1 = form_1.cleaned_data
            config_2 = form_2.cleaned_data
            results_1 = NeuralQueryView.get_most_relevant_doc_based_on_config(config_1, query_string, target_index)
            results_2 = NeuralQueryView.get_most_relevant_doc_based_on_config(config_2, query_string, target_index)

            combined_results = self.combine_results(results_1, results_2)

            # Hack. Re-adding target_index because the html page would need it for subsequent queries
            config_1["target_index"], config_2["target_index"] = target_index, target_index

            # Need to do this because we are highlighting differences between the configs in the UI
            config_1, config_2 = copy_keys_to_each_other(config_1, config_2)
            zipped_configs = list(zip(sorted(config_1.items()), sorted(config_2.items())))
            context = {
                "configs": zipped_configs,
                "model_1": config_1["reranker"],
                "model_2": config_2["reranker"],
                "query": query_string,
                "has_result": True,
                "combined_results": sorted(combined_results.items()),
                "target_index": target_index,
            }

            return render(request, self.template_name, context)
        else:
            raise Exception("Malformed query: {0}, {1}".format(form_1.errors, form_2.errors))

    @staticmethod
    def combine_results(result_1, result_2):
        combined_results = defaultdict(lambda: {"config_1_rank": None, "config_2_rank": None, "relevance": 0})

        for i, result in enumerate(result_1):
            doc_id = result["doc_id"]
            combined_results[doc_id]["config_1_rank"] = i + 1
            combined_results[doc_id]["relevance"] = result["relevance"]

        for i, result in enumerate(result_2):
            doc_id = result["doc_id"]
            combined_results[doc_id]["config_2_rank"] = i + 1
            combined_results[doc_id]["relevance"] = result["relevance"]

        return combined_results


class BM25View(TemplateView):
    template_name = "experiment.html"

    def get(self, request, *args, **kwargs):
        request_dict = request.GET.dict()
        form = BM25Form(request_dict)

        if form.is_valid():
            cleaned_data = form.cleaned_data
            results = self.get_bm25_results(
                cleaned_data["query"], cleaned_data["target_index"], cleaned_data.get("b"), cleaned_data.get("k1")
            )
            context = {
                "config": sorted(cleaned_data.items()),
                "query": request_dict["query"],
                "has_result": True,
                "results": results,
                "target_index": request_dict["target_index"],
            }
            return render(request, self.template_name, context)
        else:
            raise ValueError("Query has errors: {0}".format(form.errors))

    @staticmethod
    def get_bm25_results(query_string, target_index, b, k1):
        index_class = Index.get_index_from_index_path(target_index)
        index = index_class(target_index)

        bm25_kwargs = {"n": NUM_RESULTS_TO_SHOW}
        if b is not None:
            bm25_kwargs["b"] = b
        if k1 is not None:
            bm25_kwargs["k1"] = k1

        doc_ids, docs = BM25View.do_query(query_string, index, **bm25_kwargs)
        docs = [doc[:250] for doc in docs]
        collection = Collection.get_collection_from_index_path(index.index_path)
        relevances = collection.get_relevance(query_string, doc_ids) if collection is not None else [0] * len(doc_ids)
        result_dicts = NeuralQueryView.construct_result_dicts(doc_ids, docs, relevances)
        return result_dicts

    @staticmethod
    def do_query(query_string, index, k1=1.2, b=0.75, n=100):
        """
        Does a bm25 search and returns the most relevant 1000 docs and their ids
        """
        with patch('pyserini.setup.configure_classpath') as mock_setup:
            mock_setup.return_value = None
            from pyserini.search import pysearch
            searcher = pysearch.SimpleSearcher(index.index_path)
            searcher.set_bm25_similarity(k1, b)
            hits = searcher.search(query_string, n)
            doc_ids = [hit.docid for hit in hits]
            docs = [index.getdoc(doc_id) for doc_id in doc_ids]

            return doc_ids, docs


class CompareBM25AndNeuralQueryView(TemplateView, QueryDictParserMixin):
    template_name = "experiment_comparison.html"

    def get(self, request, *args, **kwargs):
        config_1, config_2 = self.get_two_configs_from_query_dict(request.GET)
        nir_config = config_2 if config_2["reranker"] != "BM25" else config_1
        bm25_config = config_2 if config_2["reranker"] == "BM25" else config_1

        bm25_form = BM25Form(bm25_config)
        query_string = nir_config["query"]
        target_index = nir_config["target_index"]
        nir_form_class = QueryForm.FORM_CLASSES[nir_config["reranker"]]
        nir_form = nir_form_class(nir_config)

        if bm25_form.is_valid() and nir_form.is_valid():
            bm25_data = bm25_form.cleaned_data
            bm25_results = BM25View.get_bm25_results(query_string, target_index, bm25_data.get("b"), bm25_data.get("k1"))
            nir_data = nir_form.cleaned_data

            # TODO: Make pipeline initialization robust to extra parameters. We have to remove query and target index
            # because pipeline init errors out otherwise
            nir_data.pop("target_index")
            nir_data.pop("query")
            nir_results = NeuralQueryView.get_most_relevant_doc_based_on_config(nir_data, query_string, target_index)
            combined_results = CompareNeuralQueryView.combine_results(bm25_results, nir_results)
            nir_data["target_index"] = target_index
            bm25_config = {"reranker": "BM25", "b": bm25_data["b"], "k1": bm25_data["k1"], "target_index": target_index}

            # Need to do this because we are highlighting differences between the configs in the UI
            bm25_config, nir_data = copy_keys_to_each_other(bm25_config, nir_data)
            zipped_configs = list(zip(sorted(bm25_config.items()), sorted(nir_data.items())))
            context = {
                "configs": zipped_configs,
                "model_1": "BM25",
                "model_2": nir_data["reranker"],
                "query": nir_config["query"],
                "has_result": True,
                "combined_results": sorted(combined_results.items()),
                "target_index": nir_config["target_index"],
            }

            return render(request, self.template_name, context)
        else:
            raise ValueError("Query is malformed: BM25: {0}, NIR: {1}".format(bm25_form.errors, nir_form.errors))
