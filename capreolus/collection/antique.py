import os
import shutil

from capreolus import constants
from capreolus.utils.common import download_file, hash_file
from capreolus.utils.loginit import get_logger

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class ANTIQUE(Collection):
    """A Non-factoid Question Answering Benchmark from Hashemi et al. [1]

    [1] Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W. Bruce Croft. 2020. ANTIQUE: A non-factoid question answering benchmark. ECIR 2020.
    """

    module_name = "antique"
    _path = PACKAGE_PATH / "data" / "antique-collection"

    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"

    def download_if_missing(self):
        url = "http://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt"
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
        return (
            hash_file(os.path.join(path, "antique-collection.txt"))
            == "409e0960f918970977ceab9e5b1d372f45395af25d53b95644bdc9ccbbf973da"
        )
