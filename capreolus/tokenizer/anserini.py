from capreolus import ConfigOption

from . import Tokenizer


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
