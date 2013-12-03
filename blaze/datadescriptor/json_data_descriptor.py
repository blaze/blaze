from __future__ import absolute_import
import operator
import contextlib
import ctypes
import json
import itertools as it

from .data_descriptor import IDataDescriptor
from .. import datashape
from dynd import nd, ndt
from .dynd_data_descriptor import DyNDDataDescriptor


def json_descriptor_iter(array):
    for row in array:
        yield DyNDDataDescriptor(row)

class JSONDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a JSON file.

    Parameters
    ----------
    jsonfile : file IO handle
        A file handler for the JSON file.
    schema : string or blaze.datashape
        A blaze datashape (or its string representation) of the schema
        in the JSON file.
    """
    def __init__(self, jsonfile, schema):
        if not hasattr(jsonfile, "__iter__"):
            raise TypeError('jsonfile does not have an iter interface')
        self.jsonfile = jsonfile
        if type(schema) in (str, unicode):
            schema = datashape.dshape(schema)
        self.schema = str(schema)
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

    @property
    def persistent(self):
        return True

    @property
    def is_concrete(self):
        """Returns True, JSON arrays are concrete."""
        return True

    @property
    def dshape(self):
        return datashape.dshape(self.schema)

    @property
    def writable(self):
        return False

    @property
    def appendable(self):
        return False

    @property
    def immutable(self):
        return False

    @property
    def _arr_cache(self):
        if self._cache_arr is not None:
            return self._cache_arr
        self.jsonfile.seek(0)  # go to the beginning of the file
        # This will read everything in-memory (but a memmap approach
        # is in the works)
        self._cache_arr = nd.parse_json(
            self.schema, self.jsonfile.read())
        return self._cache_arr

    def dynd_arr(self):
        return self._arr_cache

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        # Not clear to me what the length of a json object should be
        return None

    def __getitem__(self, key):
        return DyNDDataDescriptor(self._arr_cache[key])

    def __setitem__(self, key, value):
        # JSON files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        return json_descriptor_iter(self._arr_cache)

