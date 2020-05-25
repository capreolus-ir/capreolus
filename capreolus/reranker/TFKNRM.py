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

        # The extractor passed in MUST be a TFEmbedText for this to work
        self.query_embeddings = extractor.query_embeddings
        self.doc_embeddings = extractor.doc_embeddings

        self.kernels = RbfKernelBankTF(mus, sigmas, dim=1, requires_grad=config["gradkernels"])
        self.combine = tf.keras.layers.Dense(1, input_shape=(self.kernels.count(),))

    def _naive_embedding_lookup(self, embeddings, indices):
        """
        Looks up tensors from partitioned embeddings according to the supplied indices

        embeddings - A list of tensors. Each tensor in the list must occuppy less than 2GB in memory
        indices - a set of indices into the tensors. The values in indices are as if a single tensor was formed
        by stacking `embeddings` together along axis=0
        """

        num_partitions = len(embeddings)
        indices = tf.cast(indices, tf.int32)
        partition_size = tf.shape(embeddings[0])[0]
        partition_assignments = tf.cast((indices // partition_size), tf.int32)
        partition_offsets = tf.cast(indices % partition_size, tf.int32)
        partition_to_offsets = tf.dynamic_partition(partition_offsets, partition_assignments, num_partitions)
        lookups = []
        for i in range(num_partitions):
            offsets = partition_to_offsets[i]
            lookups.append(tf.gather(embeddings[i], offsets))

        return tf.concat(lookups, 0)

    def get_score(self, doc_tok, query_tok):
        query = self._naive_embedding_lookup(self.query_embeddings, query_tok[:, 0])
        doc = self._naive_embedding_lookup(self.doc_embeddings, doc_tok[:, 0])
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
