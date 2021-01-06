import faiss
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch
import os
import numpy as np
from capreolus import ConfigOption, constants, get_logger, Dependency
from capreolus.sampler import CollectionSampler

from . import Index


logger = get_logger(__name__)
faiss_logger = get_logger("faiss")


@Index.register
class FAISSIndex(Index):
    module_name = "faiss"
    
    dependencies = [Dependency(key="encoder", module="encoder", name="tinybert"), Dependency(key="index", module="index", name="anserini"), Dependency(key="benchmark", module="benchmark")] + Index.dependencies

    def _create_index(self):
        from jnius import autoclass
        anserini_index = self.index
        anserini_index.create_index()
        anserini_index_path = anserini_index.get_index_path().as_posix()
 

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(anserini_index_path).toPath())
        anserini_index_reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        
        self.encoder.build_model()

        # TODO: Figure out a better way to set this class member
        faiss_index = faiss.IndexFlatIP(self.encoder.hidden_size)

        # TODO: Add check for deleted rows in the index
        collection_docids = [anserini_index.convert_lucene_id_to_doc_id(i) for i in range(0, anserini_index_reader.maxDoc())]
        faiss_logger.debug("collection docids are like: {}".format(collection_docids[:10]))

        self.encoder.extractor.preprocess([], collection_docids, [])
        faiss_logger.error("berttext samples for faiss index creation")
        dataset = CollectionSampler()
        dataset.prepare(collection_docids, None, self.encoder.extractor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, pin_memory=True, num_workers=1
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.model.to(device)
        self.encoder.model.eval()
        faiss_logger.info("Is index trained: {}".format(faiss_index.is_trained))
        # self.doc_embs = []
        for bi, batch in tqdm(enumerate(dataloader), desc="FAISS index creation"):
            batch = {k: v.to(device) if not isinstance(v, list) else v for k, v in batch.items()}
            with torch.no_grad():
                doc_emb = self.encoder.encode(batch["posdoc"]).cpu().numpy()
            # self.doc_embs.append((batch["posdocid"], doc_emb))
            assert doc_emb.shape == (1, self.encoder.hidden_size)
            # TODO: Batch the encoding?
   
            faiss_index.add(doc_emb)
            

        faiss_logger.debug("{} docs added to FAISS index".format(faiss_index.ntotal))
        os.makedirs(self.get_index_path(), exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(self.get_index_path(), "faiss.index"))

        # TODO: write the "done" file

    def search(self, topic_vectors, k):
        faiss_logger.debug("topic_vectors shape is {}".format(topic_vectors.shape))
        # for docid, doc_emb in self.doc_embs:
            # score = F.cosine_similarity(torch.from_numpy(topic_vectors), torch.from_numpy(doc_emb))
            # faiss_logger.debug("Docid: {}, score: {}".format(docid, score))

        search_start = time.time()
        faiss_index = faiss.read_index(os.path.join(self.get_index_path(), "faiss.index"))
        faiss_logger.debug("FAISS index search took {}".format(time.time() - search_start))

        return faiss_index.search(topic_vectors, k)

    def get_docs(self, doc_ids):
        return [self.index.get_doc(doc_id) for doc_id in doc_ids]


    def get_doc(self, docid):
        return self.index.get_doc(docid)
        

