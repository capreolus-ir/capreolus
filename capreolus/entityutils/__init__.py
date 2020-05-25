from os.path import join, exists

from capreolus.registry import ModuleBase, RegisterableModule, Dependency

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

class EntityUtils(ModuleBase, metaclass=RegisterableModule):
    "the module base class"

    module_type = "entityutils"


class EntityUtilsWiki2vec(EntityUtils):
    name = "wiki2vec"

    embedding_file = "/GW/PKB/nobackup/wikipedia2vec_pretrained/enwiki_20180420_300d"
    wiki2vec = None

    def load_pretrained_emb(self):
        if self.wiki2vec is not None:
            return

        import gensim
        logger.debug("loading wikipedia2vec pretrained embedding")

        if exists(self.embedding_file + "-normed.bin"):
            self.wiki2vec = gensim.models.KeyedVectors.load(self.embedding_file + "-normed.bin", mmap='r')
            # self.wiki2vec.syn0norm = self.wiki2vec.syn0  # prevent recalc of normed vectors DEPRICATED
            self.wiki2vec.vector_norm = self.wiki2vec.vectors
        else:
            self.wiki2vec = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file + ".txt")
            self.wiki2vec.init_sims(replace=True)
            self.wiki2vec.save(self.embedding_file + "-normed.bin")

class EntityUtilsWikiLinks(EntityUtils):
    name = "wikilinks"

    wplinks_file = "/GW/PKB/work/data_personalization/entity_expansion/wikipediaInfoNeedsTypeCheck_en_20200101.tsv"
    outlinks = {}
    inlinks = {}
    total_nodes = set()

    def load_wp_links(self):
        if len(self.total_nodes) != 0:
            return
        logger.debug("Loading wikilink graph")
        with open(join(self.wplinks_file), 'r') as f:
            for line in f:
                split = line.split('\t')
                if len(split) < 4:
                    continue
                if split[2] != "<linksTo>":
                    continue

                e1 = split[1]
                e2 = split[3]

                if e1.startswith("<Category:") or e1.startswith("<Wikipedia:") or e1.startswith("<File:") or e1.startswith("<List_of"):
                    continue
                if e2.startswith("<Category:") or e2.startswith("<Wikipedia:") or e2.startswith("<File:") or e2.startswith("<List_of"):
                    continue

                self.total_nodes.add(e1)
                self.total_nodes.add(e2)

                if e1 not in self.outlinks:
                    self.outlinks[e1] = []
                self.outlinks[e1].append(e2)

                if e2 not in self.inlinks:
                    self.inlinks[e2] = []
                self.inlinks[e2].append(e1)

    def get_outlinks(self, e):
        if e not in self.outlinks:
            return []
        return self.outlinks[e]

    def get_inlinks(self, e):
        if e not in self.inlinks:
            return []
        return self.inlinks[e]

    @property
    def total_nodes_count(self):
        return len(self.total_nodes)

