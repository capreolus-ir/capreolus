from capreolus.registry import ModuleBase, RegisterableModule, Dependency
from transformers import BertTokenizer as HFBertTokenizer


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

class BertTokenizer(Tokenizer):
    name = "berttokenizer"

    @staticmethod
    def config():
        pass

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert_tokenizer = HFBertTokenizer.from_pretrained(cfg["pretrained"])

    def convert_tokens_to_ids(self, tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self.bert_tokenizer.tokenize(sentences)

        return [self.bert_tokenizer.tokenize(s) for s in sentences]
