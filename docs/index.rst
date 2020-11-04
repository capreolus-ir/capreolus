Capreolus
=========================================
Capreolus is a toolkit for constructing flexible *ad hoc retrieval pipelines*. Capreolus pipelines can be run via a Python or command line interface.

Want to jump in? `Get started with a Notebook. <https://colab.research.google.com/drive/161FnmLt3PgIXG-Z5eNg45z2iSZucVAnr?usp=sharing>`_ |Colab Badge|

Capreolus is organized around the idea of interchangeable and configurable *modules*, such as a neural ``Reranker`` or a first stage ``Searcher``. Researchers can implement new module classes, such as a new neural ``Reranker``, to experiment with a new module while controlling for all other variables in the pipeline (e.g., the first stage ranking method and its parameters, folds used for cross-validation, tokenization and embeddings if applicable used with the reranker, neural training options like the number of iterations, batch size, and loss function, etc).

Since Capreolus v0.2, *pipelines* are instances of the ``Task`` module and can be combined like any other module.
For example, the ``RerankTask`` implements a "search-then-rerank" pipeline by running ``RankTask`` and reranking its output.
Both ``Task`` modules respect the same folds (provided by a ``Benchmark``) and can be configured independently (e.g., to optimize for different metrics).

Looking for the code? `Find Capreolus on GitHub. <https://github.com/capreolus-ir/capreolus>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick
   installation
   cli
   modules
   tpu

.. Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Looking for the previous "search-then-rerank" pipeline that was presented in the WSDM'20 demo paper?
Check out Capreolus v0.1 and `the corresponding documentation. <https://capreolus.ai/en/v0.1.4/>`_

.. |Colab Badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Colab
    :scale: 100%
    :target: https://colab.research.google.com/drive/161FnmLt3PgIXG-Z5eNg45z2iSZucVAnr?usp=sharing



