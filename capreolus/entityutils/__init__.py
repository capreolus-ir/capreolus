from os.path import join
import re

import numpy as np
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
        "tokenizer": Dependency(module="tokenizer", name="anserini", config_overrides={"keepstops": False}),
    }

    embedding_dir = "/GW/PKB/nobackup/wikipedia2vec_pretrained/"
    domain_pages_dir = ''  # TODO

    @staticmethod
    def config():
        embedding = 'enwiki_20180420_300d'
        strategy = 'domain-vector-100'
        domain_relatedness_threshold = 0.3

        if not re.match(r"^(manual-domain-pages|domain-vector)-(\d+)$", strategy):
            raise ValueError(f"invalid domain embedding strategy: {strategy}")

    def initialize(self, domain):
        self.domain = domain
        self.wiki2vec = self.get_pretrained_emb()
        self.domain_rep = self.get_domain_rep()

    def get_domain_related_entities(self, entities):
        entities = ["ENTITY/{}".format(e.replace(" ", "_")) for e in entities]
        entity_vectors = [self.wiki2vec.word_vec(e) for e in entities]
        entity_similarities = self.wiki2vec.cosine_similarities(self.domain_rep, entity_vectors)
        similarities = {entities[i]: entity_similarities[i] for i in range(0, len(entities))}
        sorted_sim = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse = True)}
        print(f"Domain: {self.domain}, Strategy: {self.cfg['strategy']}, similarities:")
        print(sorted_sim)
        ret = [k for k, v in similarities.items() if v >= self.cfg['domain_relatedness_threshold']]
        return ret

    def get_pretrained_emb(self):
        import gensim

        model_path = join(self.embedding_dir, f"{self.cfg['embedding']}.txt")
        return gensim.models.KeyedVectors.load_word2vec_format(model_path)

    def get_domain_rep(self):
        m = re.match(r"^(manual-domain-pages|domain-vector)-(\d+)$", self.cfg['strategy'])
        if m:
            strategy = m.group(1)
            if m.group(1) == 'domain-vector':
                k = int(m.group(2))
            else:
                filename = self.cfg['strategy']

        if strategy == 'domain-vector':
            return self.load_domain_vector_by_neighbors(self.domain, k)
        elif strategy == 'manual-domain-pages':
            return self.load_domain_vector_by_famous_pages(self.domain, filename)

    def load_domain_vector_by_neighbors(self, domain, k):
        domain_entity = "ENTITY/Book"
        if domain == 'movie':
            domain_entity = "ENTITY/Film"
        elif domain == 'travel':
            domain_entity = "ENTITY/Travel"
        elif domain == 'food':
            domain_entity = "ENTITY/Food"

        domain_vec = self.wiki2vec.get_vector(domain_entity)
        domain_neighborhood = self.wiki2vec.most_similar(positive=[domain_vec], topn=k)
        domain_neighbors = [n[0] for n in domain_neighborhood]
        domain_rep = np.mean(self.wiki2vec[domain_neighbors], axis=0)
        return domain_rep

    def load_domain_vector_by_famous_pages(self, filename):
        with open(join(self.domain_pages_dir, f"{self.domain}_{filename}"), 'r') as f:#TODO fix format of the file
            temp = self["tokenizer"].tokenize(f.read())
            tokens = []
            for t in temp:
                if self.wiki2vec.__contains__(t):
                    tokens.append(t)
                elif self.wiki2vec.__contains__(t.lower()):
                    tokens.append(t.lower)
            domain_rep = np.mean(self.wiki2vec[tokens], axis=0)
            return domain_rep
