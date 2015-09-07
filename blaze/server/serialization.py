from collections import namedtuple
from functools import partial
import json as json_module

import pandas.msgpack as msgpack_module

from ..compatibility import pickle as pickle_module, unicode
from ..utils import json_dumps


SerializationFormat = namedtuple('SerializationFormat', 'name loads dumps')


def _coerce_str(bytes_or_str):
    if isinstance(bytes_or_str, unicode):
        return bytes_or_str
    return bytes_or_str.decode('utf-8')


json = SerializationFormat(
    'json',
    lambda data: json_module.loads(_coerce_str(data)),
    partial(json_module.dumps, default=json_dumps),
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
)


all_formats = frozenset(
    g for _, g in globals().items() if isinstance(g, SerializationFormat)
)


__all__ = ['all_formats'] + list(f.name for f in all_formats)
