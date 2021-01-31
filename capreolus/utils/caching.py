import os
import shutil

import numpy as np

# do not seed because RNG's purpose is to avoid filename conflicts
_filerng = np.random.default_rng()


class TargetFileExists(Exception):
    pass


class cached_file(object):
    def __init__(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        self.final_fn = fn

    def __enter__(self):
        if os.path.exists(self.final_fn):
            raise TargetFileExists(self.final_fn)

        self.tmp_fn = f"{self.final_fn}.tmp_{os.getpid()}_{_filerng.random()}"
        return self.tmp_fn

    def __exit__(self, extype, value, traceback):
        if extype is not None:
            # caught an exception
            os.remove(self.tmp_fn)
            return

        if os.path.exists(self.final_fn):
            raise TargetFileExists(self.final_fn)

        # TODO race condition between exists() and move(). should be safe, since contents should be deterministic?
        shutil.move(self.tmp_fn, self.final_fn)
