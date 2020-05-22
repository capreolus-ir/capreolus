import tensorflow as tf

from capreolus.reranker.common import RbfKernelBankTF, similarity_matrix_tf
from capreolus.reranker import Reranker
from capreolus.registry import Dependency
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class TFKNRM_Class(tf.keras.Model):
    def __init__(self, extractor, config, **kwargs):
        super(TFKNRM_Class, self).__init__(**kwargs)
        self.config = config
        self.extractor = extractor
        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        # self.embedding = tf.keras.layers.Embedding(
        #     len(self.extractor.stoi), self.extractor.embeddings.shape[1], weights=[self.extractor.embeddings], trainable=False
        # )
        self.kernels = RbfKernelBankTF(mus, sigmas, dim=1, requires_grad=config["gradkernels"])
        self.combine = tf.keras.layers.Dense(1, input_shape=(self.kernels.count(),))

    def get_score(self, doc_tok, query_tok):
        # query = self.embedding(query_tok)
        # doc = self.embedding(doc_tok)
        query = tf.gather(self.extractor.query_embeddings, query_tok[:, 0])
        doc = tf.gather(self.extractor.doc_embeddings, doc_tok[:, 0])

        batch_size, qlen, doclen = tf.shape(query)[0], tf.shape(query)[1], tf.shape(doc)[1]

        simmat = similarity_matrix_tf(query, doc)

        k = self.kernels(simmat)
        doc_k = tf.reduce_sum(k, axis=3)  # sum over document
        reshaped_simmat = tf.broadcast_to(
            tf.reshape(simmat, (batch_size, 1, qlen, doclen)), (batch_size, self.kernels.count(), qlen, doclen)
        )
        mask = tf.reduce_sum(reshaped_simmat, axis=3) != 0.0
        log_k = tf.where(mask, tf.math.log(doc_k + 1e-6), tf.cast(mask, tf.float32))
        query_k = tf.reduce_sum(log_k, axis=2)
        scores = self.combine(query_k)

        return tf.reshape(scores, [batch_size])

    def call(self, x, **kwargs):
        """
        During training, both posdoc and negdoc are passed
        During eval, both posdoc and negdoc are passed but negdoc would be a zero tensor
        Whether negdoc is a legit doc tensor or a dummy zero tensor is determined by which sampler is used
        (eg: sampler.TrainDataset) as well as the extractor (eg: EmbedText)

        Unlike the pytorch KNRM model, KNRMTF accepts both the positive and negative document in its forward pass.
        It scores them separately and returns the score difference (i.e posdoc_score - negdoc_score).
        """
        posdoc, negdoc, query = x[0], x[1], x[2]
        posdoc_score, negdoc_score = self.get_score(posdoc, query), self.get_score(negdoc, query)
        # During eval, the negdoc_score would be a zero tensor
        # TODO: Verify that negdoc_score is indeed always zero whenever a zero negdoc tensor is passed into it
        return posdoc_score - negdoc_score


class TFKNRM(Reranker):
    name = "TFKNRM"
    dependencies = {
        "extractor": Dependency(module="extractor", name="tfembedtext"),
        "trainer": Dependency(module="trainer", name="tensorflow"),
    }

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        finetune = False  # Fine tune the embedding

    def build(self):
        self.model = TFKNRM_Class(self["extractor"], self.cfg)
        return self.model
