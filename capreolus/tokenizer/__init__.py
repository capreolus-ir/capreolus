from profane import import_all_modules

# import_all_modules(__file__, __package__)

from profane import ModuleBase, Dependency, ConfigOption

from transformers import BertTokenizer as HFBertTokenizer


class Tokenizer(ModuleBase):
    module_type = "tokenizer"


@Tokenizer.register
class AnseriniTokenizer(Tokenizer):
    module_name = "anserini"
    config_spec = [
        ConfigOption("keepstops", True, "keep stopwords if True"),
        ConfigOption("stemmer", "none", "stemmer: porter, krovetz, or none"),
    ]

    def build(self):
        self._tokenize = self._get_tokenize_fn()

    def _get_tokenize_fn(self):
        from jnius import autoclass

        stemmer, keepstops = self.config["stemmer"], self.config["keepstops"]
        if stemmer is None:
            stemmer = "none"

        emptyjchar = autoclass("org.apache.lucene.analysis.CharArraySet").EMPTY_SET
        Analyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")
        analyzer = Analyzer.newStemmingInstance(stemmer, emptyjchar) if keepstops else Analyzer.newStemmingInstance(stemmer)
        tokenizefn = autoclass("io.anserini.analysis.AnalyzerUtils").analyze

        def _tokenize(sentence):
            return tokenizefn(analyzer, sentence).toArray()

        return _tokenize

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self._tokenize(sentences)

        return [self._tokenize(s) for s in sentences]


@Tokenizer.register
class BertTokenizer(Tokenizer):
    module_name = "berttokenizer"
    config_spec = [ConfigOption("pretrained", "bert-base-uncased", "pretrained model to load vocab from")]

    def build(self):
        self.bert_tokenizer = HFBertTokenizer.from_pretrained(self.config["pretrained"])

    def convert_tokens_to_ids(self, tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self.bert_tokenizer.tokenize(sentences)

        return [self.bert_tokenizer.tokenize(s) for s in sentences]
