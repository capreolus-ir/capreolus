import os
import shutil
import tarfile

from capreolus.registry import ModuleBase, RegisterableModule, PACKAGE_PATH
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class Collection(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "collection"
    is_large_collection = False

    def get_path_and_types(self):
        if not os.path.exists(self.path):
            self.download_if_missing()

        return self.path, self.collection_type, self.generator_type

    def download_if_missing(self):
        raise IOError(
            f"a download URL is not configured for collection={self.name} and the collection path {self.path} does not exist; you must manually place the document collection at this path in order to use this collection"
        )


class Robust04(Collection):
    name = "robust04"
    path = "/home/andrew/Aquaint-TREC-3-4"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class DummyCollection(Collection):
    name = "dummy"
    path = PACKAGE_PATH / "data" / "dummy" / "data"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class Robust05(Collection):
    name = "robust05"
    path = "missingpath"
    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"


class ANTIQUE(Collection):
    name = "antique"
    path = "/home/x978zhan/mpi-spring/data/antique/collection"

    collection_type = "TrecCollection"
    generator_type = "JsoupGenerator"

    def download_if_missing(self):
        if os.path.exists(self.path):
            return

        url = "https://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt"
        tmp_dir = os.path.join(os.path.dirname(self.path), "tmp")
        tmp_filename = os.path.join(tmp_dir, "tmp.anqique.file")
        coll_filename = os.path.join(self.path, "antique-collection.txt")

        os.makedirs(tmp_dir, exist_ok=True)
        download_file(url, tmp_filename, expected_hash=False)
        self._convert_to_trec(inp_path=tmp_filename, outp_path=coll_filename)
        logger.info(f"antique collection file prepared, stored at {coll_filename}")

        for file in os.listdir(tmp_dir):    # in case there are legacy files
            os.remove(os.path.join(tmp_dir, file))
        shutil.rmtree(tmp_dir)

    def _convert_to_trec(self, inp_path, outp_path):
        assert os.path.exists(inp_path)
        os.makedirs(os.path.dirname(outp_path), exist_ok=True)

        fout = open(outp_path, "w", encoding="utf-8")
        with open(inp_path, "r", encoding="utf-8") as f:
            for line in f:
                docid, doc = line.strip().split('\t')
                fout.write(f"<DOC>\n<DOCNO>{docid}</DOCNO>\n<TEXT>\n{doc}\n</TEXT>\n</DOC>\n")
        fout.close()
        logger.debug(f"Converted file {os.path.basename(inp_path)} to TREC format, output to: {outp_path}")