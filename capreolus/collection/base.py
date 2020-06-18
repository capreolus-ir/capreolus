import os

from capreolus import ModuleBase, Dependency, ConfigOption, constants


class Collection(ModuleBase):
    module_type = "collection"
    is_large_collection = False
    _path = None

    def get_path_and_types(self):
        if not self.validate_document_path(self._path):
            self._path = self.find_document_path()

        return self._path, self.collection_type, self.generator_type

    def validate_document_path(self, path):
        """ Attempt to validate the document collection at `path`.

            By default, this will only check whether `path` exists. Subclasses should override
            `_validate_document_path(path)` with their own logic to perform more detailed checks.

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

            We first check the collection's config for a path key. If found, `self.validate_document_path` checks
            whether the path is valid. Subclasses should override the private method `self._validate_document_path`
            with custom logic for performing checks further than existence of the directory. See `Robust04`.

            If a valid path was not found, call `download_if_missing`.
            Subclasses should override this method if downloading the needed documents is possible.

            If a valid document path cannot be found, an exception is thrown.

            Returns:
                path to this collection's raw documents
        """

        # first, see if the path was provided as a config option
        if "path" in self.config and self.validate_document_path(self.config["path"]):
            return self.config["path"]

        # if not, see if the collection can be obtained through its download_if_missing method
        return self.download_if_missing()

    def download_if_missing(self):
        raise IOError(
            f"a download URL is not configured for collection={self.module_name} and the collection path does not exist; you must manually place the document collection at this path in order to use this collection"
        )
