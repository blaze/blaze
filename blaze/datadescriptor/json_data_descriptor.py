from __future__ import absolute_import, division, print_function

import os
import json

import datashape
from ..utils import partition_all

from .data_descriptor import DDesc
from .. import py2help
from dynd import nd
from .dynd_data_descriptor import DyND_DDesc, Capabilities
from .as_py import ddesc_as_py
from .util import coerce


def json_descriptor_iter(array):
    for row in array:
        yield DyND_DDesc(row)


class JSON_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a JSON file.

    Parameters
    ----------
    path : string
        A path string for the JSON file.
    schema : string or datashape
        A datashape (or its string representation) of the schema
        in the JSON file.
    """
    def __init__(self, path, mode='r', **kwargs):
        if 'w' not in mode and os.path.isfile(path) is not True:
            raise ValueError('JSON file "%s" does not exist' % path)
        self.path = path
        self.mode = mode
        schema = kwargs.get("schema", None)
        if type(schema) in py2help._strtypes:
            schema = datashape.dshape(schema)
        self.schema = str(schema)
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

    @property
    def dshape(self):
        return datashape.dshape('var * ' + str(self.schema))

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
        with open(self.path, mode=self.mode) as jsonfile:
            # This will read everything in-memory (but a memmap approach
            # is in the works)
            self._cache_arr = nd.parse_json(
                self.schema, jsonfile.read())
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
        with open(self.path) as f:
            for line in f:
                yield coerce(self.schema, json.loads(line))

    def _iterchunks(self, blen=100):
        with open(self.path) as f:
            for chunk in partition_all(blen, f):
                text = '[' + ',\r\n'.join(chunk) + ']'
                dshape = str(len(chunk)) + ' * ' + self.schema
                yield nd.parse_json(dshape, text)

    def _extend(self, rows):
        with open(self.path, self.mode) as f:
            f.seek(0, os.SEEK_END)  # go to the end of the file
            for row in rows:
                json.dump(row, f)
                f.write('\n')

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
