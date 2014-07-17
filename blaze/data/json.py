from __future__ import absolute_import, division, print_function

import os
import json

from itertools import islice
import datashape
from dynd import nd
from collections import Iterator
from datashape.discovery import discover
from datashape import var

from ..utils import partition_all, nth, nth_list, ndget
from .. import compatibility
from ..compatibility import _inttypes, map
from .core import DataDescriptor, isdimension
from .utils import coerce, coerce_row_to_dict, coerce_to_ordered


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

    def __init__(self, path, mode='rt', schema=None, dshape=None, open=open):
        self.path = path
        self.mode = mode
        self.open = open
        if dshape:
            dshape = datashape.dshape(dshape)
        if schema:
            schema = datashape.dshape(schema)
        if dshape and not schema and isdimension(dshape[0]):
            schema = dshape.subarray(1)

        if not schema and not dshape:
            try:
                f = open(self.path, 'r')
            except:
                raise ValueError('No schema detected')
            dshape = discover(json.load(f))
            f.close()
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

        self._schema = schema
        self._dshape = dshape

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

    def as_dynd(self):
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

    def __init__(self, path, mode='rt', schema=None, dshape=None, open=open,
                 nrows_discovery=50):
        self.path = path
        self.mode = mode
        self.open = open
        if dshape:
            dshape = datashape.dshape(dshape)
        if schema:
            schema = datashape.dshape(schema)
        if dshape and not schema and isdimension(dshape[0]):
            schema = dshape.subshape[0]
        if schema and not dshape:
            dshape = var * schema

        if not schema and not dshape:
            try:
                f = open(self.path, 'r')
            except:
                raise ValueError('No schema detected')
            data = list(map(json.loads,
                            islice(f, 1, nrows_discovery)))
            f.close()
            dshape = discover(data)
            schema = dshape.subshape[0]
        # Initially the array is not loaded (is this necessary?)
        self._cache_arr = None

        self._schema = schema
        self._dshape = dshape

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

    def _get_py(self, key):
        if isinstance(key, tuple):
            result = self[key[0]]
            if isinstance(key[0], (list, slice)):
                return (ndget(key[1:], row) for row in result)
            else:
                return ndget(key[1:], result)
        f = self.open(self.path)
        if isinstance(key, _inttypes):
            result = json.loads(nth(key, f))
        elif isinstance(key, slice):
            result = map(json.loads, islice(f, key.start, key.stop, key.step))
        elif isinstance(key, list):
            result = map(json.loads, nth_list(key, f))
        else:
            raise NotImplementedError('Fancy indexing not supported')
        try:
            if not isinstance(result, Iterator):
                f.close()
            pass
        except AttributeError:
            pass
        return result

    def _iter(self):
        f = self.open(self.path)
        for line in f:
            yield json.loads(line)
        try:
            f.close()
        except AttributeError:
            pass

    __iter__ = DataDescriptor.__iter__

    def as_py(self):
        return tuple(self)

    def _iterchunks(self, blen=100):
        f = self.open(self.path)
        for chunk in partition_all(blen, f):
            text = '[' + ',\r\n'.join(chunk) + ']'
            dshape = str(len(chunk)) + ' * ' + self.schema
            yield nd.parse_json(dshape, text)
        try:
            f.close()
        except AttributeError:
            pass

    @property
    def appendable(self):
        return any(c in self.mode for c in 'wa+')

    def _extend(self, rows):
        if isinstance(self.schema[0], datashape.Record):
            transform = lambda row: coerce_row_to_dict(self.schema, row)
        else:
            transform = lambda x: x
        if not self.appendable:
            raise IOError("Read only access")
        f = self.open(self.path, self.mode)
        f.seek(0, os.SEEK_END)  # go to the end of the file
        for row in rows:
            json.dump(transform(row), f)
            f.write('\n')
        try:
            f.close()
        except AttributeError:
            pass

    def _chunks(self, blen=100):
        f = self.open(self.path)
        for chunk in partition_all(blen, f):
            text = '[' + ',\r\n'.join(chunk) + ']'
            dshape = str(len(chunk) * self.schema)
            yield nd.parse_json(dshape, text)
        try:
            f.close()
        except AttributeError:
            pass
