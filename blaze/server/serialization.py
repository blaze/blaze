from collections import namedtuple
from functools import partial
import json as json_module

import pandas.msgpack as msgpack_module

from ..compatibility import pickle as pickle_module, unicode
from ..utils import json_dumps


class SerializationFormat(namedtuple(
        'SerializationFormat', 'name loads dumps stream_unpacker')):
    def __new__(_cls, name, loads, dumps, stream_unpacker=None):
        return super(SerializationFormat, _cls).__new__(
            _cls, name, loads, dumps, stream_unpacker,
        )


def _coerce_str(bytes_or_str):
    if isinstance(bytes_or_str, unicode):
        return bytes_or_str
    return bytes_or_str.decode('utf-8')


class JsonUnpacker(object):
    def __init__(self):
        self._buffer = bytearray()

    def feed(self, next_bytes):
        self._buffer.extend(next_bytes)

    def __iter__(self):
        buffer_ = self._buffer
        loads = json_module.loads
        for n in range(len(buffer_) + 1):
            try:
                obj = loads(buffer_[:n].decode('utf-8'))
            except ValueError:
                continue

            self._buffer = buffer_ = buffer_[n:]
            yield obj


json = SerializationFormat(
    'json',
    lambda data: json_module.loads(_coerce_str(data)),
    partial(json_module.dumps, default=json_dumps),
    JsonUnpacker,
)
pickle = SerializationFormat(
    'pickle',
    pickle_module.loads,
    partial(pickle_module.dumps, protocol=pickle_module.HIGHEST_PROTOCOL),
)
msgpack = SerializationFormat(
    'msgpack',
    partial(msgpack_module.unpackb, encoding='utf-8'),
    partial(msgpack_module.packb, default=json_dumps),
    msgpack_module.Unpacker,
)


all_formats = frozenset(
    g for _, g in globals().items() if isinstance(g, SerializationFormat)
)
stream_formats = frozenset(
    f for f in all_formats if f.stream_unpacker is not None
)


__all__ = ['all_formats'] + list(f.name for f in all_formats)
