import operator
import numpy as np

from capreolus.registry import ModuleBase, RegisterableModule, Dependency

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class EntitySpecificity(ModuleBase, metaclass=RegisterableModule):
    "the module base class"

    module_type = "entityspecificity"

    def top_specific_entities(self, entities):# TODO this function is extactly repeated in another module... maybe write them in a better way or write them in a common utils...
        specificity_graph = {}
        reversed_specificity_graph = {}
        if len(entities) <= self.cfg['return_top']:
            logger.debug(f"number of entities less than top-specific-entity cut {len(entities)} <= {self.cfg['return_top']}")
            return entities
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
                if s in reversed_graph:
                    for n in reversed_graph[s]:
                        outlink_counts[n] -= 1

        return result[:self.cfg['return_top']]

class EntitySpecificityBy2HopPath(EntitySpecificity):
    name = "twohoppath"

    dependencies = {
        "popularity": Dependency(module="entitypopularity", name='centralitydegree', config_overrides={'direction': 'in'}),
        'utils': Dependency(module="entityutils", name="wikilinks")
    }

    @staticmethod
    def config():
        ranking_strategy = 'greedy_most_outlinks_withrm'
        return_top = 10

    def initialize(self):
        self['utils'].load_wp_links()
        self['popularity'].initialize()

    def rank_entity_pair_by_specificity(self, e1, e2):
        pope1 = self['popularity'].get_popularity_degree(e1)
        pope2 = self['popularity'].get_popularity_degree(e2)
        p_e1_e2 = self.prob_A_given_B_and_popA(e1, e2, pope1)
        p_e2_e1 = self.prob_A_given_B_and_popA(e2, e1, pope2)

        if p_e1_e2 > p_e2_e1:
            return [e1, e2]
        elif p_e2_e1 > p_e1_e2:
            return [e2, e1]
        else:
            logger.warning(f"equal probabilities p({e1}|{e2}) == p({e2}|{e1}).")  # this would almost never happen.
            # then return alphabetically:
            if e1 <= e2:
                return [e1, e2]
            else:
                return [e2, e1]

    # p(A| G, pop(A))
    def prob_A_given_B_and_popA(self, A, B, popA):
        A = "<{}>".format(A).replace(" ", "_")
        B = "<{}>".format(B).replace(" ", "_")
        nu = self.nodes_in_2_hop_path_from_e1_to_e2(B, A)

        B_2_hop_neighbors = self.get_2_hop_neighbors_of(B)
        de = 0
        for n in B_2_hop_neighbors:
            de += self.nodes_in_2_hop_path_from_e1_to_e2(B, n)

        res = nu / de
        res /= popA

        return res

    def nodes_in_2_hop_path_from_e1_to_e2(self, e1, e2):
        outs_e1 = self['utils'].get_outlinks(e1)
        ins_e2 = self['utils'].get_inlinks(e2)
        e1_to_e2_cnt = len(list(set(outs_e1) & set(ins_e2)))  ## not the path actually, but the number of Z in e1->Z->e2

        return e1_to_e2_cnt

    def get_2_hop_neighbors_of(self, X):
        first_hop = self['utils'].get_outlinks(X)
        second_hop = set()
        for n in first_hop:
            if n in self['utils'].outlinks:
                next_hop = self['utils'].get_outlinks(n)
                second_hop.update(next_hop)

        return list(second_hop)


class EntitySpecificityHigherMean(EntitySpecificity):
    name = 'higherneighborhoodmean'
    dependencies = {
        'utils': Dependency(module="entityutils", name="wiki2vec")
    }

    embedding_dir = "/GW/PKB/nobackup/wikipedia2vec_pretrained/"

    @staticmethod
    def config():
        k = 100
        ranking_strategy = 'greedy_most_outlinks_withrm'
        return_top = 10

    def initialize(self):
        logger.debug("loading wikipedia2vec pretrained embedding")
        self['utils'].load_pretrained_emb()

    def rank_entity_pair_by_specificity(self, e1, e2):
        w2ve1 = "ENTITY/{}".format(e1.replace(" ", "_"))
        w2ve2 = "ENTITY/{}".format(e2.replace(" ", "_"))
        if not self['utils'].wiki2vec.__contains__(w2ve1):
            logger.warning(f"entity {e1} does not exist in wiki2vec")
            return []
        if not self['utils'].wiki2vec.__contains__(w2ve2):
            logger.warning(f"entity {e2} does not exist in wiki2vec")
            return []

        e1_neighbors = self['utils'].wiki2vec.most_similar(positive=[w2ve1], topn=self.cfg["k"])
        e1_sims = [float(n[1]) for n in e1_neighbors]
        e2_neighbors = self['utils'].wiki2vec.most_similar(positive=[w2ve2], topn=self.cfg["k"])
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
