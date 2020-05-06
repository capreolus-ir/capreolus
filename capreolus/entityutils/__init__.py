import json
from os.path import join, exists
import re
import os

import numpy as np
from capreolus.utils.common import get_file_name

from capreolus.utils.loginit import get_logger

from capreolus.registry import ModuleBase, RegisterableModule, Dependency

logger = get_logger(__name__)

# in this module I want to implement entity specificity, domain relatedness, and popularity. I could have 3 different modules for them also. What's be best way?
class EntityUtils(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "entityutils"

class DomainRelatednessWiki2Vec(EntityUtils):
    name = 'relatednesswiki2vec'
    dependencies = {
        "benchmark": Dependency(module="benchmark"),
    }
    # this module probably should be dependent on entitylinking (semantically at least). Needs more thoughts.
    # At the moment, it's not and the extractor is dependant on both of them. So there we would choose which entitylinking strategy and which entity-relatedness on top of that. which is okay I guess.

    embedding_dir = "/GW/PKB/nobackup/wikipedia2vec_pretrained/"
    domain_pages_dir = ''  # TODO

    @staticmethod
    def config():
        embedding = 'enwiki_20180420_300d'
        strategy = 'domain-vector-100'
        domain_relatedness_threshold = 0.3

        if not re.match(r"^centroid-(?:entity-word-(\d+(?:\.\d+)?)-)?k(\d+)$", strategy):
            raise ValueError(f"invalid domain embedding strategy: {strategy}")

    def initialize(self, domain):
        if hasattr(self, "domain"):
            return

        self.domain = domain
        logger.debug("loading wiki2vec embeddings")

        self.wiki2vec = self.get_pretrained_emb()
        logger.debug(f"getting domain representative {self.cfg['strategy']}")
        self.domain_rep = self.get_domain_rep()

    def get_domain_related_entities(self, entities):
        entities = ["ENTITY/{}".format(e.replace(" ", "_")) for e in entities]
        entity_vectors = [self.wiki2vec.word_vec(e) for e in entities]
        entity_similarities = self.wiki2vec.cosine_similarities(self.domain_rep, entity_vectors)
        similarities = {entities[i]: entity_similarities[i] for i in range(0, len(entities))}
        sorted_sim = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse = True)}
        logger.debug(f"Domain: {self.domain}, Strategy: {self.cfg['strategy']}, similarities:")
        logger.debug(sorted_sim)
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

        domain_vec = self.wiki2vec.get_vector(domain_entity)
        domain_neighborhood = self.wiki2vec.most_similar(positive=[domain_vec], topn=k)
        domain_neighbors = [n[0] for n in domain_neighborhood]
        domain_rep = np.mean(self.wiki2vec[domain_neighbors], axis=0)
        return domain_rep
