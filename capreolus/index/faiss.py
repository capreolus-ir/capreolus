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
    
    dependencies = [Dependency(key="encoder", module="encoder", name="tinybert"), Dependency(key="index", module="index", name="anserini")] + Index.dependencies

    def _create_index(self):
        from jnius import autoclass
        anserini_index = self.index
        anserini_index._create_index()
        anserini_index_path = anserini_index.get_index_path().as_posix()
 

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

        faiss_index = faiss.IndexFlatL2(64)
        vec  = np.zeros(64)
        
        self.encoder.build_model()

        for i in range(0, anserini_index_reader.maxDoc()):
            # TODO: Add check for deleted rows
            # TODO: Batch the encoding?
            doc = anserini_index_reader.document(i)
            print(dir(doc))
            doc_contents = doc.content
            doc_vector = self.encoder.encode(doc_contents)
            faiss_index.add(doc_vector)
            

        logger.error("{} docs added to FAISS index".format(faiss_index.ntotal))
        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(self.get_index_path(), "faiss.index"))

        # TODO: write the "done" file


    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]


    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

