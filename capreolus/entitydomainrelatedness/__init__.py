import re
import time
from os.path import exists, join
import os
import json

import numpy as np

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, PACKAGE_PATH
from capreolus.utils.common import get_file_name

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class EntityDomainRelatedness(ModuleBase, metaclass=RegisterableModule):
    "the module base class"

    module_type = "entitydomainrelatedness"


class DomainRelatedness(EntityDomainRelatedness):
    name = 'wiki2vecrepresentative'
    default_settings_dir = PACKAGE_PATH / "data" / "domain_entity_relatedness_setting"
    setting_file_name_mapping = {"book_prCacc": "book_maxPRAUC_maxACC",
                                 "food_prCacc": "food_maxPRAUC_maxACC",
                                 "travel_wikivoyage_prCacc": "travel_wikivoyage_maxPRAUC_maxACC",
                                 "movie_prCacc": "movie_maxPRAUC_maxACC"}
    dependencies = {
        "benchmark": Dependency(module="benchmark"),
        'utils': Dependency(module="entityutils", name="wiki2vec")
    }

    strategy_NE = None
    strategy_C = None
    domain_relatedness_threshold_NE = None
    domain_relatedness_threshold_C = None

    @staticmethod
    def config():
        # if you want to give other settings and thresholds as input, pass the file name and put the file in the default_settings_dir
        strategy_NE = None
        strategy_C = None
        domain_relatedness_threshold_NE = None
        domain_relatedness_threshold_C = None

        if strategy_NE is not None:
            if not re.match("(book|food|travel_wikivoyage|movie)_prCacc", strategy_NE):
                raise ValueError(f"invalid strategy_NE {strategy_NE}")

        if strategy_C is not None:
            if not re.match("(book|food|travel_wikivoyage|movie)_prCacc", strategy_C):
                raise ValueError(f"invalid strategy_C {strategy_C}")

        if domain_relatedness_threshold_NE is not None:
            if not re.match("(book|food|travel_wikivoyage|movie)_prCacc", domain_relatedness_threshold_NE):
                raise ValueError(f"invalid domain_relatedness_threshold_NE {domain_relatedness_threshold_NE}")

        if domain_relatedness_threshold_C is not None:
            if not re.match("(book|food|travel_wikivoyage|movie)_prCacc", domain_relatedness_threshold_C):
                raise ValueError(f"invalid domain_relatedness_threshold_C {domain_relatedness_threshold_C}")

        return_top = -1

    def e_strategy(self, isNE):
        if isNE:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_NE)
        else:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_C)
        if m:
            if not m.group(5):
                k = 0
                m = None
            else:
                k = int(m.group(5))
                m = m.group(6)

        return k, m

    def d_strategy(self, isNE):
        if isNE:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_NE)
        else:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_C)
        if m:
            if not m.group(2):
                k = 0
                m = None
            else:
                k = int(m.group(2))
                m = m.group(3)

        return k, m

    def get_similarities_cache_path(self):
        # logger.debug(self.entity_linking_cache_path / "similarities")
        return self.entity_linking_cache_path / "similarities"

    def initialize(self, el_cache_path):
        if hasattr(self, "domain"):
            return
        self.domain = self["benchmark"].domain
        self.entity_linking_cache_path = el_cache_path
        self['utils'].load_pretrained_emb()

        if self.cfg['strategy_NE'] is None or self.cfg['strategy_C'] is None or self.cfg['domain_relatedness_threshold_NE'] is None or self.cfg['domain_relatedness_threshold_NE'] is None:
            raise ValueError(f"strategies or thresholds should not be None")

        self.strategy_NE = json.load(open(join(self.default_settings_dir, self.setting_file_name_mapping[self.cfg['strategy_NE']]), 'r'))['strategy_NE']
        self.strategy_C = json.load(open(join(self.default_settings_dir, self.setting_file_name_mapping[self.cfg['strategy_C']]), 'r'))['strategy_C']
        if self.strategy_NE is not None and not re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_NE):
            raise ValueError(f"invalid domain embedding strategyNE: {self.strategy_NE}")
        if self.strategy_C is not None and not re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.strategy_C):
            raise ValueError(f"invalid domain embedding strategyC: {self.strategy_C}")


        self.domain_relatedness_threshold_NE = float(json.load(open(join(self.default_settings_dir, self.setting_file_name_mapping[self.cfg['domain_relatedness_threshold_NE']]), 'r'))['domain_relatedness_threshold_NE'])
        self.domain_relatedness_threshold_C = float(json.load(open(join(self.default_settings_dir, self.setting_file_name_mapping[self.cfg['domain_relatedness_threshold_C']]), 'r'))['domain_relatedness_threshold_C'])

        k, m = self.d_strategy(True)
        self.domain_rep_NE = self.load_domain_vector_by_neighbors(k, m)
        k, m = self.d_strategy(False)
        self.domain_rep_C = self.load_domain_vector_by_neighbors(k, m)

        self.entity_rep_cache = {"C": {}, "NE": {}}

    def calculate_domain_entity_similarities(self, entities, isNE):
        logger.debug("calculating similarity between domain model and extracted entities")

        ek, em = self.e_strategy(isNE)
        corne = "NE" if isNE else "C"

        entities_in_w2v = []
        entity_vectors = []
        for e in entities:
            w2ve = "ENTITY/{}".format(e.replace(" ", "_"))
            if self['utils'].wiki2vec.__contains__(w2ve):
                entities_in_w2v.append(e)
                if w2ve not in self.entity_rep_cache[corne]:
                    self.entity_rep_cache[corne][w2ve] = self.get_representative(w2ve, ek, em)
                entity_vectors.append(self.entity_rep_cache[corne][w2ve])
                entity_vectors.append(self['utils'].wiki2vec.word_vec(w2ve))

        if len(entity_vectors) == 0:
            return {}

        if isNE:
            domain_rep = self.domain_rep_NE
        else:
            domain_rep = self.domain_rep_C
        entity_similarities = self['utils'].wiki2vec.cosine_similarities(domain_rep, entity_vectors)
        similarities = {entities_in_w2v[i]: float(entity_similarities[i]) for i in range(0, len(entities_in_w2v))}
        for e in entities:
            if e not in similarities:
                similarities[e] = -1

        return similarities

    def get_domain_related_entities(self, tid, entities):
        #load from the cached similarities if exists else calculate them
        benchmark_name = self['benchmark'].name
        benchmark_querytype = self['benchmark'].query_type

        outdir = self.get_similarities_cache_path()
        if exists(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype))):
            logger.debug(f"Sim folder: {outdir}" )
            similarities = json.load(open(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype)), 'r'))
        else:
            similaritiesNE = self.calculate_domain_entity_similarities(entities["NE"], True)
            similaritiesC = self.calculate_domain_entity_similarities(entities["C"], False)
            similarities = {"NE": similaritiesNE, "C": similaritiesC}
            os.makedirs(outdir, exist_ok=True)
            with open(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype)), 'w') as f:
                sorted_sim = {}
                sorted_sim['NE'] = {k: v for k, v in sorted(similarities['NE'].items(), key=lambda item: item[1], reverse=True)}
                sorted_sim['C'] = {k: v for k, v in sorted(similarities['C'].items(), key=lambda item: item[1], reverse=True)}
                f.write(json.dumps(sorted_sim, sort_keys=True, indent=4))

        # now thresholding
        if self.cfg['return_top'] == -1:
            NEs = [k for k, v in similarities["NE"].items() if v >= self.domain_relatedness_threshold_NE]
            Cs = [k for k, v in similarities["C"].items() if v >= self.domain_relatedness_threshold_C]
            return {"NE": NEs, "C": Cs}

        trNEs = {k:v for k, v in similarities["NE"].items() if v >= self.domain_relatedness_threshold_NE}
        trCs = {k:v for k, v in similarities["C"].items() if v >= self.domain_relatedness_threshold_C}
        retNE = [k for k,v in sorted(trNEs.items(), key=lambda item: item[1], reverse=True)]
        retC = [k for k, v in sorted(trCs.items(), key=lambda item: item[1], reverse=True)]
        return {"NE": retNE[:self.cfg['return_top']], "C": retC[:self.cfg['return_top']]}

    def load_domain_vector_by_neighbors(self, k, method):
        if self.domain == "book":
            domain_entity = "ENTITY/Book"
        elif self.domain == "movie":
            domain_entity = "ENTITY/Film"
        elif self.domain == "travel_wikivoyage":
            domain_entity = "ENTITY/Travel"
        elif self.domain == "food":
            domain_entity = "ENTITY/Food"

        return self.get_representative(domain_entity, k, method)

    def get_representative(self, initial, k, method):
        init_vec = self['utils'].wiki2vec.get_vector(initial)
        if k == 0:
            return init_vec
        neighborhood = self['utils'].wiki2vec.most_similar(positive=[init_vec], topn=k)
        neighbors = []
        weights = []
        for n in neighborhood:
            neighbors.append(n[0])
            weights.append(n[1])

        if method == 'avg':
            rep = np.average(self['utils'].wiki2vec[neighbors], axis=0)
        elif method == 'wavg':
            rep = np.average(self['utils'].wiki2vec[neighbors], axis=0, weights=weights)
        # elif method == 'max':
        #     rep = np.max(self['utils'].wiki2vec[neighbors], axis=0)
        else:
            raise ValueError(f"vector aggregation method not implemented or typo: {method}")

        return rep