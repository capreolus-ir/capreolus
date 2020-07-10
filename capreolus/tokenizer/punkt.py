from capreolus import constants

from . import Tokenizer


@Tokenizer.register
class PunktTokenizer(Tokenizer):
    module_name = "punkt"
    _tokenizer = None

    def tokenize(self, txt):
        if self._tokenizer is None:
            import nltk

            nltk_dir = constants["CACHE_BASE_PATH"] / "nltk"
            nltk.download("punkt", download_dir=nltk_dir, quiet=True)
            self._tokenizer = nltk.data.load((nltk_dir / "tokenizers/punkt/english.pickle").as_posix())

        if not txt:
            return []

        return self._tokenizer.tokenize(txt)
