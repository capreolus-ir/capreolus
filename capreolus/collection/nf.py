import os
import tarfile

from capreolus import constants, get_logger
from capreolus.utils.common import download_file

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class NF(Collection):
    """ NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval [1]

    [1] Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler. A Full-Text Learning to Rank Dataset for Medical Information Retrieval Proceedings of the 38th European Conference on Information Retrieval (ECIR), Padova, Italy, 2016. https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/
    """

    module_name = "nf"
    _path = PACKAGE_PATH / "data" / "nf-collection"
    url = "http://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz"

    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"

    def download_raw(self):
        cachedir = self.get_cache_path()
        tmp_dir = cachedir / "tmp"
        tmp_tar_fn, tmp_corpus_dir = tmp_dir / "nfcorpus.tar.gz", tmp_dir / "nfcorpus"

        os.makedirs(tmp_dir, exist_ok=True)

        if not tmp_tar_fn.exists():
            download_file(self.url, tmp_tar_fn, "ebc026d4a8bef3f866148b727e945a2073eb4045ede9b7de95dd50fd086b4256")

        with tarfile.open(tmp_tar_fn) as f:
            f.extractall(tmp_dir)
        return tmp_corpus_dir

    def download_if_missing(self):
        cachedir = self.get_cache_path()
        document_dir = os.path.join(cachedir, "documents")
        coll_filename = os.path.join(document_dir, "nf-collection.txt")
        if os.path.exists(coll_filename):
            return document_dir

        os.makedirs(document_dir, exist_ok=True)
        tmp_corpus_dir = self.download_raw()

        inp_fns = [tmp_corpus_dir / f"{set_name}.docs" for set_name in ["train", "dev", "test"]]
        print(inp_fns)
        with open(coll_filename, "w", encoding="utf-8") as outp_file:
            self._convert_to_trec(inp_fns, outp_file)
        logger.info(f"nf collection file prepared, stored at {coll_filename}")

        return document_dir

    def _convert_to_trec(self, inp_fns, outp_file):
        # train.docs, dev.docs, and test.docs have some overlap, so we check for duplicate docids
        seen_docids = set()

        for inp_fn in inp_fns:
            assert os.path.exists(inp_fn)

            with open(inp_fn, "rt", encoding="utf-8") as f:
                for line in f:
                    docid, doc = line.strip().split("\t")

                    if docid not in seen_docids:
                        outp_file.write(f"<DOC>\n<DOCNO>{docid}</DOCNO>\n<TEXT>\n{doc}\n</TEXT>\n</DOC>\n")
                        seen_docids.add(docid)
