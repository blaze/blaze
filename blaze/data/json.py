from __future__ import absolute_import, division, print_function

import os
import json

from itertools import islice
import datashape
from dynd import nd

from ..utils import partition_all, nth
from .. import py2help
from ..py2help import _inttypes
from .core import DataDescriptor
from .utils import coerce

def isdimension(ds):
    return isinstance(ds, (datashape.Var, datashape.Fixed))


class JSON(DataDescriptor):
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
    immutable = True
    deferred = False
    persistent = True
    appendable = False
    remote = False

    def __init__(self, path, mode='r', schema=None, dshape=None, open=open):
        self.path = path
        self.mode = mode
        self.open = open
        if dshape:
            dshape = datashape.dshape(dshape)
        if dshape and not schema and isdimension(dshape[0]):
            schema = dshape.subarray(1)

        if isinstance(schema, py2help._strtypes):
            schema = datashape.dshape(schema)
        if not schema:
            # TODO: schema detection from file
            raise ValueError('No schema found')
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

        self._schema = schema
        self._dshape = dshape

    @property
    def dshape(self):
        return self._dshape or datashape.dshape('var * ' + str(self.schema))

    @property
    def _arr_cache(self):
        if self._cache_arr is not None:
            return self._cache_arr
        jsonfile = self.open(self.path)
        # This will read everything in-memory (but a memmap approach
        # is in the works)
        self._cache_arr = nd.parse_json(str(self.dshape), jsonfile.read())
        try:
            jsonfile.close()
        except:
            pass
        return self._cache_arr

    def __iter__(self):
        for line in self._arr_cache:
            yield nd.as_py(line)

    def dynd_arr(self):
        return self._arr_cache

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)


class JSON_Streaming(JSON):
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
    immutable = False

    @property
    def _arr_cache(self):
        if self._cache_arr is not None:
            return self._cache_arr
        jsonfile = self.open(self.path)
        # This will read everything in-memory (but a memmap approach
        # is in the works)
        text = '[' + ', '.join(jsonfile) + ']'
        try:
            jsonfile.close()
        except:
            pass
        self._cache_arr = nd.parse_json(str(self.dshape), text)
        return self._cache_arr

    def __getitem__(self, key):
        with self.open(self.path) as f:
            if isinstance(key, _inttypes):
                result = json.loads(nth(key, f))
            elif isinstance(key, slice):
                result = list(map(json.loads,
                                    islice(f, key.start, key.stop, key.step)))
            else:
                raise NotImplementedError('Fancy indexing not supported\n'
                        'Create DyND array and use fancy indexing from there')
        return coerce(self.schema, result)

    def _iter(self):
        with self.open(self.path) as f:
            for line in f:
                yield json.loads(line)

    def _iterchunks(self, blen=100):
        with self.open(self.path) as f:
            for chunk in partition_all(blen, f):
                text = '[' + ',\r\n'.join(chunk) + ']'
                dshape = str(len(chunk)) + ' * ' + self.schema
                yield nd.parse_json(dshape, text)

    @property
    def appendable(self):
        return any(c in self.mode for c in 'wa+')

    def _extend(self, rows):
        if not self.appendable:
            raise IOError("Read only access")
        with self.open(self.path, self.mode) as f:
            f.seek(0, os.SEEK_END)  # go to the end of the file
            for row in rows:
                json.dump(row, f)
                f.write('\n')

    def _chunks(self, blen=100):
        with self.open(self.path) as f:
            for chunk in partition_all(blen, f):
                text = '[' + ',\r\n'.join(chunk) + ']'
                dshape = str(len(chunk) * self.schema)
                yield nd.parse_json(dshape, text)
