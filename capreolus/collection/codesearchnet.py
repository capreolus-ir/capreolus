import pickle
from zipfile import ZipFile

from tqdm import tqdm

from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class CodeSearchNet(Collection):
    """CodeSearchNet Corpus. [1]

       [1] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. arXiv 2019.
    """

    module_name = "codesearchnet"
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    collection_type = "TrecCollection"  # TODO: any other supported type?
    generator_type = "DefaultLuceneDocumentGenerator"
    config_spec = [ConfigOption("lang", "ruby", "CSN language dataset to use")]

    def download_if_missing(self):
        cachedir = self.get_cache_path()
        document_dir = cachedir / "documents"
        coll_filename = document_dir / ("csn-" + self.config["lang"] + "-collection.txt")

        if coll_filename.exists():
            return document_dir

        zipfile = self.config["lang"] + ".zip"
        lang_url = f"{self.url}/{zipfile}"
        tmp_dir = cachedir / "tmp"
        zip_path = tmp_dir / zipfile

        if zip_path.exists():
            logger.info(f"{zipfile} already exist under directory {tmp_dir}, skip downloaded")
        else:
            tmp_dir.mkdir(exist_ok=True, parents=True)
            download_file(lang_url, zip_path)

        document_dir.mkdir(exist_ok=True, parents=True)  # tmp
        with ZipFile(zip_path, "r") as zipobj:
            zipobj.extractall(tmp_dir)

        pkl_path = tmp_dir / (self.config["lang"] + "_dedupe_definitions_v2.pkl")
        self._pkl2trec(pkl_path, coll_filename)
        return document_dir

    def _pkl2trec(self, pkl_path, trec_path):
        lang = self.config["lang"]
        with open(pkl_path, "rb") as f:
            codes = pickle.load(f)

        fout = open(trec_path, "w", encoding="utf-8")
        for i, code in tqdm(enumerate(codes), desc=f"Preparing the {lang} collection file"):
            docno = f"{lang}-FUNCTION-{i}"
            doc = remove_newline(" ".join(code["function_tokens"]))
            fout.write(document_to_trectxt(docno, doc))
        fout.close()
