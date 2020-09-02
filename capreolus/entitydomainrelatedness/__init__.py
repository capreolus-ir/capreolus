import re
import time
from os.path import exists, join
import os
import json

import numpy as np

from capreolus.registry import ModuleBase, RegisterableModule, Dependency
from capreolus.utils.common import get_file_name

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class EntityDomainRelatedness(ModuleBase, metaclass=RegisterableModule):
    "the module base class"

    module_type = "entitydomainrelatedness"


class DomainRelatedness(EntityDomainRelatedness):
    name = 'wiki2vecrepresentative'

    dependencies = {
        "benchmark": Dependency(module="benchmark"),
        'utils': Dependency(module="entityutils", name="wiki2vec")
    }

    @staticmethod
    def config():
        # todo if these were None, I will initialize it in the initializer based on the domain
        strategy_NE = None
        strategy_C = None
        domain_relatedness_threshold_NE = None
        domain_relatedness_threshold_C = None

        return_top = -1

        if strategy_NE is not None and not re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", strategy_NE):
            raise ValueError(f"invalid domain embedding strategyNE: {strategy_NE}")
        if strategy_C is not None and not re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", strategy_C):
            raise ValueError(f"invalid domain embedding strategyC: {strategy_C}")

    def e_strategy(self, isNE):
        if isNE:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.cfg['strategy_NE'])
        else:
            m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", self.cfg['strategy_C'])
        if m:
            if not m.group(5):
                k = 0
                m = None
            else:
                k = int(m.group(5))
                m = m.group(6)

        return k, m

    def get_similarities_cache_path(self):
        # logger.debug(self.entity_linking_cache_path / "similarities")
        while exists(self.entity_linking_cache_path / "similarities" / "lock"):
            time.sleep(10)
        with open(self.entity_linking_cache_path / "similarities" / "lock") as f:
            f.write("")
        return self.entity_linking_cache_path / "similarities"

    def initialize(self, el_cache_path):
        if hasattr(self, "domain"):
            return
        self.domain = self["benchmark"].domain
        self.entity_linking_cache_path = el_cache_path
        self['utils'].load_pretrained_emb()

        #TODO
        #have to be given from somewhere else!!!!! read frol file maybe, or set in config, buttttt we have to give the domain as well...
        # if self.cfg['strategy_NE'] is None:
        #     if self.domain == 'book':
        #         self.cfg['strategy_NE'] = "d-k:100-avg_e-k:100-wavg"
        #     elif self.domain == 'movie':
        #         self.cfg['strategy_NE'] = "d-k:0_e-k:100-avg"
        #     elif self.domain == 'food':
        #         self.cfg['strategy_NE'] = "d-k:10-avg_e-k:0"
        #     elif self.domain == 'travel':
        #         self.cfg['strategy_NE'] = "d-k:25-avg_e-k:10-avg"
        # if self.cfg['domain_relatedness_threshold_NE'] is None:
        #     if self.domain == 'book':
        #         self.cfg['domain_relatedness_threshold_NE'] = 0.57247805
        #     elif self.domain == 'movie':
        #         self.cfg['domain_relatedness_threshold_NE'] = 0.26852363
        #     elif self.domain == 'food':
        #         self.cfg['domain_relatedness_threshold_NE'] = 0.29420701
        #     elif self.domain == 'travel':
        #         self.cfg['domain_relatedness_threshold_NE'] = 0.35691148
        # logger.debug(f"getting domain representative (NE) {self.cfg['strategy_NE']}")
        self.domain_rep_NE = self.get_domain_rep(self.cfg['strategy_NE'])
        #
        # if self.cfg['strategy_C'] is None:
        #     if self.domain == 'book':
        #         self.cfg['strategy_C'] = "d-k:5-avg_e-k:100-avg"
        #     elif self.domain == 'movie':
        #         self.cfg['strategy_C'] = "d-k:5-avg_e-k:25-wavg"
        #     elif self.domain == 'food':
        #         self.cfg['strategy_C'] = "d-k:25-wavg_e-k:50-avg"
        #     elif self.domain == 'travel':
        #         self.cfg['strategy_C'] = "d-k:100-avg_e-k:0"
        # if self.cfg['domain_relatedness_threshold_C'] is None:
        #     if self.domain == 'book':
        #         self.cfg['domain_relatedness_threshold_C'] = 0.50608605
        #     elif self.domain == 'movie':
        #         self.cfg['domain_relatedness_threshold_C'] = 0.46772834
        #     elif self.domain == 'food':
        #         self.cfg['domain_relatedness_threshold_C'] = 0.38735285
        #     elif self.domain == 'travel':
        #         self.cfg['domain_relatedness_threshold_C'] = 0.57309896
        # logger.debug(f"getting domain representative (C) {self.cfg['strategy_C']}")
        self.domain_rep_C = self.get_domain_rep(self.cfg['strategy_C'])

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
            NEs = [k for k, v in similarities["NE"].items() if v >= self.cfg['domain_relatedness_threshold_NE']]
            Cs = [k for k, v in similarities["C"].items() if v >= self.cfg['domain_relatedness_threshold_C']]
            return {"NE": NEs, "C": Cs}

        trNEs = {k:v for k, v in similarities["NE"].items() if v >= self.cfg['domain_relatedness_threshold_NE']}
        trCs = {k:v for k, v in similarities["C"].items() if v >= self.cfg['domain_relatedness_threshold_C']}
        retNE = [k for k,v in sorted(trNEs.items(), key=lambda item: item[1], reverse=True)]
        retC = [k for k, v in sorted(trCs.items(), key=lambda item: item[1], reverse=True)]
        return {"NE": retNE[:self.cfg['return_top']], "C": retC[:self.cfg['return_top']]}


    def get_domain_rep(self, strategy):
        if strategy is None:
            raise RuntimeError("domain-entity relatedness strategy is None")

        m = re.match(r"^d-k:(0|(100|50|25|10|5)-(w?avg))_e-k:(0|(100|50|25|10|5)-(w?avg))$", strategy)
        if m:
            if not m.group(2):
                dk = 0
                dm = None
            else:
                dk = int(m.group(2))
                dm = m.group(3)

            return self.load_domain_vector_by_neighbors(dk, dm)

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

        if method == 'average':
            rep = np.average(self['utils'].wiki2vec[neighbors], axis=0)
        elif method == 'weighted_average':
            rep = np.average(self['utils'].wiki2vec[neighbors], axis=0, weights=weights)
        # elif method == 'max':
        #     rep = np.max(self['utils'].wiki2vec[neighbors], axis=0)
        else:
            print(f"vector aggregation method not implemented or typo: {method}")
            return

        return rep