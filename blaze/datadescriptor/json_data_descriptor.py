from __future__ import absolute_import, division, print_function

import os

import datashape

from .data_descriptor import DDesc
from .. import py2help
from dynd import nd
from .dynd_data_descriptor import DyND_DDesc, Capabilities


def json_descriptor_iter(array):
    for row in array:
        yield DyND_DDesc(row)

def all_files_in_dir(a_path):
    import os
    for root, dirs, files in os.walk(a_path):
        for f in files:
            yield os.path.join(root,f)

class Paths(object):
    """
    A small class that provides generator functionality over a bunch of files.
    If you pass in something with files (either a list of files or a directory
    containing a list of files) the Paths object will be iterable over those
    files. If not, we assume we just have one file and make an iterable over that.
    """
    def __init__(self, paths):
        try:
            # If it's a file, make it iterable
            import os
            if (os.path.isfile(paths)):
                self._paths = [paths]
            elif (os.path.isdir(paths)):
                self._paths = all_files_in_dir(paths)
            else:
                # test for iterability
                _ = (p for p in paths)
                self._paths = paths
        except TypeError:
            try:
                # test for iterability
                _ = (p for p in paths)
                self._paths = paths
            except TypeError:
                raise ValueError("Don't know what kind of data this is!")

    def __iter__(self):
        for p in self._paths:
            yield  p


class JSON_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a JSON file.

    Parameters
    ----------
    pth : a filename, an iterable of filenames, or a directory
    schema : string or datashape
        A datashape (or its string representation) of the schema
        in the JSON file.
    """
    def __init__(self, pth, mode='r', **kwargs):
        self.file_end = False
        #Set the paths iterator
        self._paths_iter = iter(Paths(pth))
        self.cur_path = self.get_next_file()
        if os.path.isfile(self.cur_path) is not True:
            raise ValueError('JSON file "%s" does not exist' % self.cur_path)
        self.mode = mode
        schema = kwargs.get("schema", None)
        if type(schema) in py2help._strtypes:
            schema = datashape.dshape(schema)
        self.schema = str(schema)
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

    @property
    def dshape(self):
        return datashape.dshape(self.schema)

    def get_next_file(self):
        try:
            return self._paths_iter.next()
        except StopIteration:
            self.file_end = True
            return None

    @property
    def capabilities(self):
        """The capabilities for the json data descriptor."""
        return Capabilities(
            # json datadescriptor cannot be updated
            immutable = False,
            # json datadescriptors are concrete
            deferred = False,
            # json datadescriptor is persistent
            persistent = True,
            # json datadescriptor can be appended efficiently
            appendable = True,
            remote = False,
            )

    @property
    def _arr_cache(self):
        if self._cache_arr is not None:
            return self._cache_arr
        if (self.cur_path):
            with open(self.cur_path, mode=self.mode) as jsonfile:
                # This will read everything in-memory (but a memmap approach
                # is in the works)
                self._cache_arr = nd.parse_json(
                    self.schema, jsonfile.read())
            #Set up the next file, if there is one
            self.cur_path = self.get_next_file()

        return self._cache_arr

    def dynd_arr(self):
        return self._arr_cache

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        # Not clear to me what the length of a json object should be
        return None

    def __getitem__(self, key):
        return DyND_DDesc(self._arr_cache[key])

    def __setitem__(self, key, value):
        # JSON files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        while (self._arr_cache is not None):
            for row in self._arr_cache:
                yield DyND_DDesc(row)
            #done with that cache, reset
            self._cache_arr = None

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
