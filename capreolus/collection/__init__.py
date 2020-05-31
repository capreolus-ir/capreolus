import os
import math
import shutil
import pickle
import tarfile
import filecmp

from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
import pandas as pd

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.common import download_file, hash_file, remove_newline
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs, document_to_trectxt

logger = get_logger(__name__)


class Collection(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "collection"
    is_large_collection = False
    _path = None

    def get_path_and_types(self):
        if not self.validate_document_path(self._path):
            self._path = self.find_document_path()

        return self._path, self.collection_type, self.generator_type

    def validate_document_path(self, path):
        """ Attempt to validate the document collection at `path`.

            By default, this will only check whether `path` exists. Subclasses should override
            `_validate_document_path(path)` with their own logic to perform more detailed checks.

            Returns:
                True if the path is valid following the logic described above, or False if it is not
         """

        if not (path and os.path.exists(path)):
            return False

        return self._validate_document_path(path)

    def _validate_document_path(self, path):
        """ Collection-specific logic for validating the document collection path. Subclasses should override this.

            Returns:
                this default method provided by Collection always returns true
         """

        return True

    def find_document_path(self):
        """ Find the location of this collection's documents (i.e., the raw document collection).

            We first check the collection's config for a path key. If found, `self.validate_document_path` checks
            whether the path is valid. Subclasses should override the private method `self._validate_document_path`
            with custom logic for performing checks further than existence of the directory. See `Robust04`.

            If a valid path was not found, call `download_if_missing`.
            Subclasses should override this method if downloading the needed documents is possible.

            If a valid document path cannot be found, an exception is thrown.

            Returns:
                path to this collection's raw documents
        """

        # first, see if the path was provided as a config option
        if "path" in self.cfg and self.validate_document_path(self.cfg["path"]):
            return self.cfg["path"]

        # if not, see if the collection can be obtained through its download_if_missing method
        return self.download_if_missing()

    def download_if_missing(self):
        raise IOError(
            f"a download URL is not configured for collection={self.name} and the collection path does not exist; you must manually place the document collection at this path in order to use this collection"
        )


class Robust04(Collection):
    name = "robust04"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"
    config_keys_not_in_path = ["path"]

    @staticmethod
    def config():
        path = "Aquaint-TREC-3-4"

    def download_if_missing(self):
        return self.download_index(
            url="https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz",
            sha256="dddb81f16d70ea6b9b0f94d6d6b888ed2ef827109a14ca21fd82b2acd6cbd450",
            index_directory_inside="index-robust04-20191213/",
            # this string should match how the index was built (i.e., Anserini, stopwords removed, Porter stemming)
            index_cache_path_string="index-anserini_indexstops-False_stemmer-porter",
            index_expected_document_count=528_030,
            cachedir=self.get_cache_path(),
        )

    def _validate_document_path(self, path):
        """ Validate that the document path appears to contain robust04's documents (Aquaint-TREC-3-4).

            Validation is performed by looking for four directories (case-insensitive): `FBIS`, `FR94`, `FT`, and `LATIMES`.
            These directories may either be at the root of `path` or they may be in `path/NEWS_data` (case-insensitive).

            Returns:
                True if the Aquaint-TREC-3-4 document directories are found or False if not
        """

        if not os.path.isdir(path):
            return False

        contents = {fn.lower(): fn for fn in os.listdir(path)}
        if "news_data" in contents:
            contents = {fn.lower(): fn for fn in os.listdir(os.path.join(path, contents["news_data"]))}

        if "fbis" in contents and "fr94" in contents and "ft" in contents and "latimes" in contents:
            return True

        return False

    def download_index(
        self, cachedir, url, sha256, index_directory_inside, index_cache_path_string, index_expected_document_count
    ):
        # Download the collection from URL and extract into a path in the cache directory.
        # To avoid re-downloading every call, we create an empty '/done' file in this directory on success.
        done_file = os.path.join(cachedir, "done")
        document_dir = os.path.join(cachedir, "documents")

        # already downloaded?
        if os.path.exists(done_file):
            return document_dir

        # 1. Download and extract Anserini index to a temporary location
        tmp_dir = os.path.join(cachedir, "tmp_download")
        archive_file = os.path.join(tmp_dir, "archive_file")
        os.makedirs(document_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info("downloading index for missing collection %s to temporary file %s", self.name, archive_file)
        download_file(url, archive_file, expected_hash=sha256)

        logger.info("extracting index to %s (before moving to correct cache path)", tmp_dir)
        with tarfile.open(archive_file) as tar:
            tar.extractall(path=tmp_dir)

        extracted_dir = os.path.join(tmp_dir, index_directory_inside)
        if not (os.path.exists(extracted_dir) and os.path.isdir(extracted_dir)):
            raise ValueError(f"could not find expected index directory {extracted_dir} in {tmp_dir}")

        # 2. Move index to its correct location in the cache
        index_dir = os.path.join(cachedir, index_cache_path_string, "index")
        if not os.path.exists(os.path.join("index_dir", "done")):
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
            shutil.move(extracted_dir, index_dir)

        # 3. Extract raw documents from the Anserini index to document_dir
        anserini_index_to_trec_docs(index_dir, document_dir, index_expected_document_count)

        # remove temporary files and create a /done we can use to verify extraction was successful
        shutil.rmtree(tmp_dir)
        with open(done_file, "wt") as outf:
            print("", file=outf)

        return document_dir


class DummyCollection(Collection):
    name = "dummy"
    _path = PACKAGE_PATH / "data" / "dummy" / "data"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    def _validate_document_path(self, path):
        """ Validate that the document path contains `dummy_trec_doc` """
        return "dummy_trec_doc" in os.listdir(path)


class ANTIQUE(Collection):
    name = "antique"
    _path = PACKAGE_PATH / "data" / "antique-collection"

    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    def download_if_missing(self):
        url = "https://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt"
        cachedir = self.get_cache_path()
        document_dir = os.path.join(cachedir, "documents")
        coll_filename = os.path.join(document_dir, "antique-collection.txt")

        if os.path.exists(coll_filename):
            return document_dir

        tmp_dir = cachedir / "tmp"
        tmp_filename = os.path.join(tmp_dir, "tmp.anqique.file")

        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(document_dir, exist_ok=True)

        download_file(url, tmp_filename, expected_hash="68b6688f5f2668c93f0e8e43384f66def768c4da46da4e9f7e2629c1c47a0c36")
        self._convert_to_trec(inp_path=tmp_filename, outp_path=coll_filename)
        logger.info(f"antique collection file prepared, stored at {coll_filename}")

        for file in os.listdir(tmp_dir):  # in case there are legacy files
            os.remove(os.path.join(tmp_dir, file))
        shutil.rmtree(tmp_dir)

        return document_dir

    def _convert_to_trec(self, inp_path, outp_path):
        assert os.path.exists(inp_path)

        fout = open(outp_path, "wt", encoding="utf-8")
        with open(inp_path, "rt", encoding="utf-8") as f:
            for line in f:
                docid, doc = line.strip().split("\t")
                fout.write(f"<DOC>\n<DOCNO>{docid}</DOCNO>\n<TEXT>\n{doc}\n</TEXT>\n</DOC>\n")
        fout.close()
        logger.debug(f"Converted file {os.path.basename(inp_path)} to TREC format, output to: {outp_path}")

    def _validate_document_path(self, path):
        """ Checks that the sha256sum is correct """
        return hash_file(path) == "409e0960f918970977ceab9e5b1d372f45395af25d53b95644bdc9ccbbf973da"


class MSMarco(Collection):
    name = "msmarco"
    config_keys_not_in_path = ["path"]
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    @staticmethod
    def config():
        path = "/GW/NeuralIR/nobackup/msmarco/trec_format"


class CodeSearchNet(Collection):
    name = "codesearchnet"
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    collection_type = "TrecCollection"  # TODO: any other supported type?
    generator_type = "JsoupGenerator"

    @staticmethod
    def config():
        lang = "ruby"

    def download_if_missing(self):
        cachedir = self.get_cache_path()
        document_dir = cachedir / "documents"
        coll_filename = document_dir / ("csn-" + self.cfg["lang"] + "-collection.txt")

        if coll_filename.exists():
            return document_dir

        zipfile = self.cfg["lang"] + ".zip"
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

        pkl_path = tmp_dir / (self.cfg["lang"] + "_dedupe_definitions_v2.pkl")
        self._pkl2trec(pkl_path, coll_filename)
        return document_dir

    def _pkl2trec(self, pkl_path, trec_path):
        lang = self.cfg["lang"]
        with open(pkl_path, "rb") as f:
            codes = pickle.load(f)

        fout = open(trec_path, "w", encoding="utf-8")
        for i, code in tqdm(enumerate(codes), desc=f"Preparing the {lang} collection file"):
            docno = f"{lang}-FUNCTION-{i}"
            doc = remove_newline(" ".join(code["function_tokens"]))
            fout.write(document_to_trectxt(docno, doc))
        fout.close()


class COVID(Collection):
    name = "covid"
    url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_%s.tar.gz"
    generator_type = "Cord19Generator"

    @staticmethod
    def config():
        coll_type = "abstract"  # "fulltext"  # options: abstract, fulltext, paragraph
        round = 3  # 3 / 2 / 1

    def __init__(self, cfg):
        super().__init__(cfg)
        coll_type, round = cfg["coll_type"], cfg["round"]
        type2coll = {
            "abstract": "Cord19AbstractCollection",
            "fulltext": "Cord19FullTextCollection",
            "paragraph": "Cord19ParagraphCollection",
        }
        dates = ["2020-04-10", "2020-05-01", "2020-05-19"]

        if coll_type not in type2coll:
            raise ValueError(f"Unexpected coll_type: {coll_type}; expeced one of: {' '.join(type2coll.keys())}")
        if round > len(dates):
            raise ValueError(f"Unexpected round number: {round}; only {len(dates)} number of rounds are provided")

        self.collection_type = type2coll[coll_type]
        self.date = dates[round - 1]

    def download_if_missing(self):
        cachedir = self.get_cache_path()
        tmp_dir, document_dir = Path("/tmp"), cachedir / "documents"
        expected_fns = [document_dir / "metadata.csv", document_dir / "document_parses"]
        if all([os.path.exists(f) for f in expected_fns]):
            return document_dir

        url = self.url % self.date
        tar_file = tmp_dir / f"covid-19-{self.date}.tar.gz"
        if not tar_file.exists():
            download_file(url, tar_file)

        with tarfile.open(tar_file) as f:
            f.extractall(path=cachedir)  # emb.tar.gz, metadata.csv, doc.tar.gz, changelog
            os.rename(cachedir / self.date, document_dir)

        doc_fn = "document_parses"
        if f"{doc_fn}.tar.gz" in os.listdir(document_dir):
            with tarfile.open(document_dir / f"{doc_fn}.tar.gz") as f:
                f.extractall(path=document_dir)
        else:
            self.transform_metadata(document_dir)

        # only document_parses and metadata.csv are expected
        for fn in os.listdir(document_dir):
            if (document_dir / fn) not in expected_fns:
                os.remove(document_dir / fn)
        return document_dir

    def transform_metadata(self, root_path):
        """
        the transformation is necessary for dataset round 1 and 2 according to
        https://discourse.cord-19.semanticscholar.org/t/faqs-about-cord-19-dataset/94

        the assumed directory under root_path:
        ./root_path
            ./metadata.csv
            ./comm_use_subset
            ./noncomm_use_subset
            ./custom_license
            ./biorxiv_medrxiv
            ./archive

        In a nutshell:
        1. renaming:
            Microsoft Academic Paper ID -> mag_id;
            WHO #Covidence -> who_covidence_id
        2. update:
            has_pdf_parse -> pdf_json_files  # e.g. document_parses/pmc_json/PMC125340.xml.json
            has_pmc_xml_parse -> pmc_json_files
        """
        metadata_csv = str(root_path / "metadata.csv")
        orifiles = ["arxiv", "custom_license", "biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset"]
        for fn in orifiles:
            if (root_path / fn).exists():
                continue

            tar_fn = root_path / f"{fn}.tar.gz"
            if not tar_fn.exists():
                continue

            with tarfile.open(str(tar_fn)) as f:
                f.extractall(path=root_path)
                os.remove(tar_fn)

        metadata = pd.read_csv(metadata_csv, header=0)
        columns = metadata.columns.values
        cols_before = [
            'cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time',
            'authors', 'journal', 'Microsoft Academic Paper ID', 'WHO #Covidence', 'arxiv_id', 'has_pdf_parse',
            'has_pmc_xml_parse', 'full_text_file', 'url']
        assert all(columns == cols_before)

        # step 1: rename column
        cols_to_rename = {"Microsoft Academic Paper ID": "mag_id", "WHO #Covidence": "who_covidence_id"}
        metadata.columns = [cols_to_rename.get(c, c) for c in columns]

        # step 2: parse path & move json file
        doc_outp = root_path / "document_parses"
        pdf_dir, pmc_dir = doc_outp / "pdf_json", doc_outp / "pmc_json"
        pdf_dir.mkdir(exist_ok=True, parents=True)
        pmc_dir.mkdir(exist_ok=True, parents=True)

        new_cols = ["pdf_json_files", "pmc_json_files"]
        for col in new_cols:
            metadata[col] = ""
        metadata["s2_id"] = math.nan  # tmp, what's this column??

        iterbar = tqdm(desc="transforming data", total=len(metadata))
        for i, row in metadata.iterrows():
            dir = row["full_text_file"]

            if row["has_pmc_xml_parse"]:
                name = row["pmcid"] + ".xml.json"
                ori_fn = root_path / dir / "pmc_json" / name
                pmc_fn = f"document_parses/pmc_json/{name}"
                metadata.at[i, "pmc_json_files"] = pmc_fn
                pmc_fn = root_path / pmc_fn
                if not pmc_fn.exists():
                    os.rename(ori_fn, pmc_fn)  # check
            else:
                metadata.at[i, "pmc_json_files"] = math.nan

            if row["has_pdf_parse"]:
                shas = str(row["sha"]).split(";")
                pdf_fn_final = ""
                for sha in shas:
                    name = sha.strip() + ".json"
                    ori_fn = root_path / dir / "pdf_json" / name
                    pdf_fn = f"document_parses/pdf_json/{name}"
                    pdf_fn_final = f"{pdf_fn_final};{pdf_fn}" if pdf_fn_final else pdf_fn
                    pdf_fn = root_path / pdf_fn
                    if not pdf_fn.exists():
                        os.rename(ori_fn, pdf_fn)  # check
                    else:
                        if ori_fn.exists():
                            assert filecmp.cmp(ori_fn, pdf_fn)
                            os.remove(ori_fn)

                metadata.at[i, "pdf_json_files"] = pdf_fn_final
            else:
                metadata.at[i, "pdf_json_files"] = math.nan

            iterbar.update()

        # step 3: remove deprecated columns, remove unwanted directories
        cols_to_remove = ["has_pdf_parse", "has_pmc_xml_parse", "full_text_file"]
        metadata.drop(columns=cols_to_remove)

        dir_to_remove = ["comm_use_subset", "noncomm_use_subset", "custom_license", "biorxiv_medrxiv", "arxiv"]
        for dir in dir_to_remove:
            dir = root_path / dir
            for subdir in os.listdir(dir):
                os.rmdir(dir / subdir)  # since we are supposed to move away all the files
            os.rmdir(dir)

        # assert len(metadata.columns) == 19
        # step 4: save back
        metadata.to_csv(metadata_csv, index=False)
