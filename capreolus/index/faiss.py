from capreolus import ConfigOption, constants, get_logger

from . import Index


logger = get_logger(__name__)


@Index.register
class FAISSIndex(Index):
    module_name = "faiss"
    
    dependencies = [Dependency(key="encoder", module="encoder"), Dependency(key="index", module="index")] + Index.dependencies

    def _create_index(self):
        collection_path = self.collection.get_path_and_types()
        # 1. Create the anserini index from collection
        # 2. Pass each document in the index through the encoder to get the vector rep
        #   a. There's some scope for batching - it would be extremely slow otherwise
        #   b. We'll call "self.encoder.encode(doc)" here in a for-loop
        # 3. Map the string doc-id to an int64 id (FAISS can handle only integer ids). This would create a one-to-one map between vectors in the FAISS index and the raw docs in anserini
        # 4. Create a faiss index and add the generated vectors to it
        # 5. Put the "done" file in the index path so that caching works


    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]


    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

