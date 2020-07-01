import os

from capreolus import ModuleBase


class Collection(ModuleBase):
    """Base class for Collection modules. The purpose of a Collection is to describe a document collection's location and its format. 

    Determining the document collection's location on disk:
        - The *path* config option will be used if it contains a valid loation.
        - If not, the ``_path`` attribute is used if it is valid. This is primarily used with :class:`~.dummy.DummyCollection`.
        - If not, the class' ``download_if_missing`` method will be called.

    Modules should provide:
        - the ``collection_type`` and ``generator_type`` class attributes, corresponding to Anserini types
        - a ``download_if_missing`` method, if the collection is publicly available
        - a ``_validate_document_path`` method. See :func:`~capreolus.collection.Collection.validate_document_path`.
    """

    module_type = "collection"
    is_large_collection = False
    _path = None

    def get_path_and_types(self):
        """ Returns a ``(path, collection_type, generator_type)`` tuple. """
        if not self.validate_document_path(self._path):
            self._path = self.find_document_path()

        return self._path, self.collection_type, self.generator_type

    def validate_document_path(self, path):
        """ Attempt to validate the document collection at ``path``.

            By default, this will only check whether ``path`` exists. Subclasses should override
            ``_validate_document_path(path)`` with their own logic to perform more detailed checks.

            Returns:
                True if the path is valid following the logic described above, or False if it is not
         """

        if not (path and os.path.exists(path)):
            return False

        return self._validate_document_path(path)

    def _validate_document_path(self, path):
        """ Collection-specific logic for validating the document collection path. Subclasses should override this.

            Returns:
                this default method provided by Collection always returns true
         """

        return True

    def find_document_path(self):
        """ Find the location of this collection's documents (i.e., the raw document collection).

            We first check the collection's config for a path key. If found, ``self.validate_document_path`` checks
            whether the path is valid. Subclasses should override the private method ``self._validate_document_path``
            with custom logic for performing checks further than existence of the directory.

            If a valid path was not found, call ``download_if_missing``.
            Subclasses should override this method if downloading the needed documents is possible.

            If a valid document path cannot be found, an exception is thrown.

            Returns:
                path to this collection's raw documents
        """

        # first, see if the path was provided as a config option
        if "path" in self.config and self.validate_document_path(self.config["path"]):
            return self.config["path"]

        # see if the path is hardcoded (e.g., for the dummy collection")
        if self._path and self.validate_document_path(self._path):
            return self._path

        # if not, see if the collection can be obtained through its download_if_missing method
        return self.download_if_missing()

    def download_if_missing(self):
        """ Download the collection and return its path. Subclasses should override this. """
        raise IOError(
            f"a download URL is not configured for collection={self.module_name} and the collection path does not exist; you must manually place the document collection at this path in order to use this collection"
        )


from profane import import_all_modules

from .dummy import DummyCollection
from .antique import ANTIQUE
from .nf import NF
from .robust04 import Robust04

import_all_modules(__file__, __package__)
