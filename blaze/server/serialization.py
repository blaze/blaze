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


class OutOfData(StopIteration):
    """An exception used to indicate that a lazy unpacker does not have
    enough data to return an object.
    """
    pass


class JsonUnpacker(object):
    """A class for consuming a stream of bytes yielding json objects as they
    are read.

    Objects may be extracted with either the ``unpack`` method or the
    iteration protocol.

    Parameters
    ----------
    buffer_ : bytearray, optional
        The initial buffer to consume from.
    """
    def __init__(self, buffer_=None):
        self._buffer = bytearray() if buffer_ is None else buffer_

    def feed(self, next_bytes):
        """Feed new data into the unpacker.

        Parameters
        ----------
        next_bytes : bytes
            New bytes to append to our internal buffer.
        """
        self._buffer.extend(next_bytes)

    def unpack(self):
        """Consume one json object from our buffer.

        Returns
        -------
        obj : any
            The first json object in our buffer if any exists.

        Raises
        ------
        OutOfData
            If there is not enough data in the buffer to return an object.
        """
        buffer_ = self._buffer
        loads = json_module.loads
        for n in range(len(buffer_) + 1):
            try:
                obj = loads(buffer_[:n].decode('utf-8'))
            except ValueError:
                continue

            self._buffer = buffer_ = buffer_[n:]
            return obj
        else:
            raise OutOfData()

    def __iter__(self):
        unpack = self.unpack
        while True:
            try:
                yield unpack()
            except OutOfData:
                return


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
    partial(msgpack_module.Unpacker, encoding='utf-8'),
)


all_formats = frozenset(
    g for _, g in globals().items() if isinstance(g, SerializationFormat)
)
stream_formats = frozenset(
    f for f in all_formats if f.stream_unpacker is not None
)


__all__ = ['all_formats'] + list(f.name for f in all_formats)
