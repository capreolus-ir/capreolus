Capreolus
=========================================

Capreolus is a toolkit for conducting end-to-end *ad hoc retrieval* experiments, which consist of a first stage ranking method (e.g., BM25 or RM3) followed by a neural re-ranking method.

Capreolus is organized around the idea of interchangeable and configurable *modules*, such as a neural ``reranker`` or a first stage ``searcher``. Researchers can implement new module classes, such as a new neural ``reranker``, to experiment with a new module while controlling for all other variables in the pipeline (e.g., the first stage ranking method and its parameters, folds used for cross-validation, tokenization and embeddings if applicable used with the reranker, neural training options like the number of iterations, batch size, and loss function, etc).

Looking for the code? `Find Capreolus on GitHub. <https://github.com/capreolus-ir/capreolus>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   cli
   experiments-wsdm20
   api
   reproduceability

..   
  TODO Concepts: explanation of modules (index, reranker, etc) and file formats (for extending)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
