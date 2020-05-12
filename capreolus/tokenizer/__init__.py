from capreolus.registry import ModuleBase, RegisterableModule, Dependency


class Tokenizer(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "tokenizer"


class AnseriniTokenizer(Tokenizer):
    name = "anserini"

    @staticmethod
    def config():
        keepstops = True
        stemmer = "none"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tokenize = self._get_tokenize_fn()

    def _get_tokenize_fn(self):
        from jnius import autoclass

        stemmer, keepstops = self.cfg["stemmer"], self.cfg["keepstops"]
        emptyjchar = autoclass("org.apache.lucene.analysis.CharArraySet").EMPTY_SET
        Analyzer = autoclass("io.anserini.analysis.EnglishStemmingAnalyzer")
        analyzer = Analyzer(stemmer, emptyjchar) if keepstops else Analyzer(stemmer)
        tokenizefn = autoclass("io.anserini.analysis.AnalyzerUtils").tokenize

        def _tokenize(sentence):
            return tokenizefn(analyzer, sentence).toArray()

        return _tokenize

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self._tokenize(sentences)

        return [self._tokenize(s) for s in sentences]

class SpacyTokenizer(Tokenizer):
    name = "spacy"

    @staticmethod
    def config():
        keepstops = True
        removesmallerlen = -1
        # stemmer = "none" # does spacy has stemmer also?

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tokenize = self._get_tokenize_fn()

    def _get_tokenize_fn(self):
        import spacy
        nlp = spacy.load('en_core_web_sm')#TODO is sm good? (also: I added spacy into requirements but, this should also be downloaded, it should be added somewhere for setup)

        keepstops = self.cfg["keepstops"]
        removesmallerlen = self.cfg["removesmallerlen"]

        def _tokenize(sentences):
            tokens = []
            for doc in nlp.pipe(sentences, disable=["ner"]):
                for token in doc:
                    if not keepstops and token.is_stop:
                        continue
                    if token.text in [" ", "-"] or token.is_punct:
                        continue
                    if removesmallerlen != -1 and len(token.text) < removesmallerlen: #todo I used this to clean the profiles
                        continue
                    tokens.append(token.text.lower()) # TODO: I lowercased everything, we could try this with anserini later since spacy is much slower
            return tokens

        return _tokenize

    def tokenize(self, sentences):#TODO (ask): where does the split to the sentences happen?
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self._tokenize([sentences])

        return [self._tokenize(s) for s in sentences]
