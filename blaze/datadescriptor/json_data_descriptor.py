from __future__ import absolute_import, division, print_function

import os
import json

from itertools import islice
import datashape
from ..utils import partition_all, nth

from .data_descriptor import DDesc
from .. import py2help
from dynd import nd
from .dynd_data_descriptor import DyND_DDesc
from .as_py import ddesc_as_py
from .util import coerce
from ..py2help import _inttypes

def isdimension(ds):
    return isinstance(ds, (datashape.Var, datashape.Fixed))


class JSON_DDesc(DDesc):
    """
    A Blaze data descriptor to expose a JSON file.

    Parameters
    ----------
    path : string
        A path string for the JSON file.
    schema : string or datashape
        A datashape (or its string representation) of the schema
        in the JSON file.
    """
    def __init__(self, path, mode='r', **kwargs):
        if 'w' not in mode and not os.path.isfile(path):
            raise ValueError('JSON file "%s" does not exist' % path)
        self.path = path
        self.mode = mode
        schema = kwargs.get("schema")
        dshape = kwargs.get('dshape')
        if dshape:
            dshape = datashape.dshape(dshape)
        if dshape and not schema and isdimension(dshape[0]):
            schema = dshape[1:]

        if isinstance(schema, py2help._strtypes):
            schema = datashape.dshape(schema)
        if not schema:
            # TODO: schema detection from file
            raise ValueError('No schema found')
        # TODO: should store DataShape object
        self.schema = str(schema)
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None
        self._dshape = dshape

    @property
    def dshape(self):
        return self._dshape or datashape.dshape('var * ' + str(self.schema))

    @property
    def capabilities(self):
        """The capabilities for the json data descriptor."""
        return {'immutable': False,
                'deferred': False,
                'persistent': True,
                'appendable': True,
                'remote': False}

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

    def __getitem__(self, key):
        return self._arr_cache[key]

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

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


class JSON_Streaming_DDesc(JSON_DDesc):
    """
    A Blaze data descriptor to expose a Streaming JSON file.

    Parameters
    ----------
    path : string
        A path string for the JSON file.
    schema : string or datashape
        A datashape (or its string representation) of the schema
        in the JSON file.
    """
    @property
    def _arr_cache(self):
        if self._cache_arr is not None:
            return self._cache_arr
        with open(self.path, mode=self.mode) as jsonfile:
            # This will read everything in-memory (but a memmap approach
            # is in the works)
            text = '[' + ', '.join(map(json.loads, jsonfile)) + ']'
            self._cache_arr = nd.parse_json(self.schema, text)
        return self._cache_arr

    def __getitem__(self, key):
        with open(self.path) as f:
            if isinstance(key, _inttypes):
                result = json.loads(nth(key, f))
            elif isinstance(key, slice):
                result = list(map(json.loads,
                                    islice(f, key.start, key.stop, key.step)))
            else:
                raise NotImplementedError('Fancy indexing not supported\n'
                        'Create DyND array and use fancy indexing from there')
        return coerce(self.schema, result)

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
