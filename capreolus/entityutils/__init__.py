import json
import operator
from os.path import join, exists
import re
import os

import numpy as np
from capreolus.utils.common import get_file_name

from capreolus.utils.loginit import get_logger

from capreolus.registry import ModuleBase, RegisterableModule, Dependency

logger = get_logger(__name__)

# in this module I want to implement entity specificity, domain relatedness, and popularity. I could have 3 different modules for them also. What's be best way?
class EntityUtilsWiki2vec(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "entityutilswiki2vec"

    embedding_dir = "/GW/PKB/nobackup/wikipedia2vec_pretrained/"
    wiki2vec = {}

    def get_pretrained_emb(self, embeddingfilename):
        if embeddingfilename in self.wiki2vec:
            return

        import gensim

        model_path = join(self.embedding_dir, f"{embeddingfilename}.txt")
        self.wiki2vec[embeddingfilename] = gensim.models.KeyedVectors.load_word2vec_format(model_path)

class DomainRelatedness(EntityUtilsWiki2vec):
    name = 'domainrelatedness'
    dependencies = {
        "benchmark": Dependency(module="benchmark"),
    }
    # this module probably should be dependent on entitylinking (semantically at least). Needs more thoughts.
    # At the moment, it's not and the extractor is dependant on both of them. So there we would choose which entitylinking strategy and which entity-relatedness on top of that. which is okay I guess.


    @staticmethod
    def config():
        embedding = 'enwiki_20180420_300d'
        strategy = 'centroid-k100'
        domain_relatedness_threshold = 0.4

        if not re.match(r"^centroid-(?:entity-word-(\d+(?:\.\d+)?)-)?k(\d+)$", strategy):
            raise ValueError(f"invalid domain embedding strategy: {strategy}")

    def get_similarities_cache_path(self):
        return self.get_cache_path() / "similarities"

    def initialize(self):
        if hasattr(self, "domain"):
            return

        self.domain = self["benchmark"].domain
        logger.debug("loading wikipedia2vec pretrained embedding")
        self.get_pretrained_emb(self.cfg['embedding'])
        logger.debug(f"getting domain representative {self.cfg['strategy']}")
        self.domain_rep = self.get_domain_rep()

    def calculate_domain_entity_similarities(self, entities):
        entities_in_w2v = []
        entity_vectors = []
        for e in entities:
            w2ve = "ENTITY/{}".format(e.replace(" ", "_"))
            if self.wiki2vec[self.cfg['embedding']].__contains__(w2ve):
                entities_in_w2v.append(e)
                entity_vectors.append(self.wiki2vec[self.cfg['embedding']].word_vec(w2ve))

        if len(entity_vectors) == 0:
            return {}

        entity_similarities = self.wiki2vec[self.cfg['embedding']].cosine_similarities(self.domain_rep, entity_vectors)
        similarities = {entities_in_w2v[i]: float(entity_similarities[i]) for i in range(0, len(entities_in_w2v))}
        for e in entities:
            if e not in similarities:
                similarities[e] = -1

        return similarities

    def get_domain_related_entities(self, tid, entities):
        #load from the cached similarities if exists else calculate them
        benchmark_name = self['benchmark'].name
        benchmark_querytype = self['benchmark'].query_type

        logger.debug("calculating similarity between domain model and extracted entities")
        outdir = self.get_similarities_cache_path()
        if exists(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype))):
            similarities = json.load(open(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype)), 'r'))
        else:
            self.initialize()
            similarities = self.calculate_domain_entity_similarities(entities)
            os.makedirs(outdir, exist_ok=True)
            with open(join(outdir, get_file_name(tid, benchmark_name, benchmark_querytype)), 'w') as f:
                f.write(json.dumps(similarities, sort_keys=True, indent=4))

        # just for logging:
        #sorted_sim = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}
        #logger.debug(f"Domain: {self.domain}, Strategy: {self.cfg['strategy']}")
        #logger.debug(f"Similarities: {sorted_sim}")

        ret = [k for k, v in similarities.items() if v >= self.cfg['domain_relatedness_threshold']]
        return ret

    def get_pretrained_emb(self):
        import gensim

        model_path = join(self.embedding_dir, f"{self.cfg['embedding']}.txt")
        return gensim.models.KeyedVectors.load_word2vec_format(model_path)

    def get_domain_rep(self):
        m = re.match(r"^centroid-(?:entity-word-(\d+(?:\.\d+)?)-)?k(\d+)$", self.cfg['strategy'])
        if m:
            k = int(m.group(2))
            if m.group(1):
                raise NotImplementedError("domain model as combination of entity neighbors and word neighbors is not implemented")
            else:
                return self.load_domain_vector_by_neighbors(k)

    def load_domain_vector_by_neighbors(self, k):
        if self.domain == "book":
            domain_entity = "ENTITY/Book"
        elif self.domain == "movie":
            domain_entity = "ENTITY/Film"
        elif self.domain == "travel_wikivoyage":
            domain_entity = "ENTITY/Travel"
        elif self.domain == "food":
            domain_entity = "ENTITY/Food"

        domain_vec = self.wiki2vec[self.cfg['embedding']].get_vector(domain_entity)
        domain_neighborhood = self.wiki2vec[self.cfg['embedding']].most_similar(positive=[domain_vec], topn=k)
        domain_neighbors = [n[0] for n in domain_neighborhood]
        domain_rep = np.mean(self.wiki2vec[self.cfg['embedding']][domain_neighbors], axis=0)
        return domain_rep


class EntitySpecificityHigherMean(EntityUtilsWiki2vec):
    name = 'specificityhighermean'
    dependencies = {
        "benchmark": Dependency(module="benchmark"),
    }

    embedding_dir = "/GW/PKB/nobackup/wikipedia2vec_pretrained/"

    @staticmethod
    def config():
        embedding = 'enwiki_20180420_300d'
        k = 100
        ranking_strategy = 'greedy_most_outlinks_withrm'
        return_top = 10

    def initialize(self):
        logger.debug("loading wikipedia2vec pretrained embedding")
        self.get_pretrained_emb(self.cfg['embedding'])

    def top_specific_entities(self, entities):
        self.initialize()

        specificity_graph = {}
        reversed_specificity_graph = {}
        logger.debug(f"number of entities: {len(entities)}")
        for i in range(0, len(entities)):
            for j in range(i+1, len(entities)):
                temp = self.rank_entity_pair_by_specificity(entities[i], entities[j])
                if len(temp) == 2:
                    if temp[0] not in specificity_graph:
                        specificity_graph[temp[0]] = []
                    specificity_graph[temp[0]].append(temp[1])

                    if temp[1] not in reversed_specificity_graph:
                        reversed_specificity_graph[temp[1]] = []
                    reversed_specificity_graph[temp[1]].append(temp[0])
        if self.cfg['ranking_strategy'] == 'greedy_most_outlinks_withrm':
            return self.rank_by_most_outlinks(specificity_graph, reversed_specificity_graph, True)
        elif self.cfg['ranking_strategy'] == 'greedy_most_outlinks_withoutrm':
            return self.rank_by_most_outlinks(specificity_graph, reversed_specificity_graph, False)
        else:
            raise NotImplementedError(f"ranking strategy {self.cfg['ranking_strategy']} to rank specific nodes not implemented")

    def rank_by_most_outlinks(self, graph, reversed_graph, remove):
        outlink_counts = {}
        for n in graph:
            outlink_counts[n] = len(graph[n])

        result = []
        while len(result) < len(graph):
            s = max(outlink_counts.items(), key=operator.itemgetter(1))[0]
            result.append(s)
            outlink_counts[s] = -1

            if remove:
                for n in reversed_graph[s]:
                    outlink_counts[n] -= 1

        return result[:self.cfg['return_top']]


    def rank_entity_pair_by_specificity(self, e1, e2):
        w2ve1 = "ENTITY/{}".format(e1.replace(" ", "_"))
        w2ve2 = "ENTITY/{}".format(e2.replace(" ", "_"))
        if not self.wiki2vec[self.cfg['embedding']].__contains__(w2ve1):
            logger.warning(f"entity {e1} does not exist in wiki2vec")
            return []
        if not self.wiki2vec[self.cfg['embedding']].__contains__(w2ve2):
            logger.warning(f"entity {e2} does not exist in wiki2vec")
            return []

        e1_neighbors = self.wiki2vec[self.cfg['embedding']].most_similar(positive=[w2ve1], topn=self.cfg["k"])
        e1_sims = [float(n[1]) for n in e1_neighbors]
        e2_neighbors = self.wiki2vec[self.cfg['embedding']].most_similar(positive=[w2ve2], topn=self.cfg["k"])
        e2_sims = [float(n[1]) for n in e2_neighbors]

        m1, m2 = np.mean(e1_sims), np.mean(e2_sims)

        if m1 > m2:
            return [e1, e2]
        elif m2 > m1:
            return [e2, e1]
        else:
            logger.warning("Neighborhood mean of entities are equal.")#this would almost never happen.
            #then return alphabetically:
            if e1 <= e2:
                return [e1, e2]
            else:
                return [e2, e1]

#TODO: instead of one module type entityutils, we could have different types: domain-relatedness, specificity, popularity
#OR embeddingmethods and linkmethoods. then I could maybe load the wiki2vec once only