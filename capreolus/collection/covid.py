import filecmp
import math
import os
import tarfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class COVID(Collection):
    """ The COVID-19 Open Research Dataset (https://www.semanticscholar.org/cord19) """

    module_name = "covid"
    url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_%s.tar.gz"
    generator_type = "Cord19Generator"
    config_spec = [ConfigOption("coll_type", "abstract", "one of: abstract, fulltext, paragraph"), ConfigOption("round", 3)]

    def build(self):
        coll_type, round = self.config["coll_type"], self.config["round"]
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
            "cord_uid",
            "sha",
            "source_x",
            "title",
            "doi",
            "pmcid",
            "pubmed_id",
            "license",
            "abstract",
            "publish_time",
            "authors",
            "journal",
            "Microsoft Academic Paper ID",
            "WHO #Covidence",
            "arxiv_id",
            "has_pdf_parse",
            "has_pmc_xml_parse",
            "full_text_file",
            "url",
        ]
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
