import faiss
import os
import numpy as np
from pyserini.index import IndexReader
from capreolus import ConfigOption, constants, get_logger, Dependency

from . import Index


logger = get_logger(__name__)


@Index.register
class FAISSIndex(Index):
    module_name = "faiss"
    
    dependencies = [Dependency(key="encoder", module="encoder", name="gloveavg"), Dependency(key="index", module="index", name="anserini")] + Index.dependencies

    def _create_index(self):
        from jnius import autoclass
        anserini_index = self.index
        anserini_index._create_index()
        anserini_index_path = anserini_index.get_index_path().as_posix()
 

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        index = faiss.IndexFlatL2(64)
        vec  = np.zeros(64)

        for i in range(0, index_reader.maxDoc()):
            # TODO: Add check for deleted rows
            # TODO: Batch the encoding?
            # 1. Get the ith doc
            # 2. Encode the ith doc
            # 3. Add the ith doc to Index
            pass

        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(index, os.path.join(self.get_index_path(), "faiss.index"))

        # TODO: write the "done" file


    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]


    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

