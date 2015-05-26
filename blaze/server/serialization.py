from collections import namedtuple
from functools import partial
import json as json_module

from ..compatibility import pickle, unicode
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
    pickle.loads,
    partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL),
)
