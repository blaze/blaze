import sys
from functools import partial
import json as json_module

from datashape.predicates import iscollection, istabular, isscalar
from odo import odo
import pandas as pd
from pandas.compat import BytesIO
from pandas.io.packers import (
    unpack as msgpack_unpack,
    decode as pd_msgpack_object_hook,
    encode as pd_msgpack_default,
)
import pandas.msgpack as msgpack_module
from toolz import identity

from ..compatibility import pickle as pickle_module, unicode, PY2
from ..interactive import coerce_scalar
from ..utils import json_dumps, object_hook


class SerializationFormat(object):
    """A serialization format for the blaze server and blaze client.

    Parameters
    ----------
    name : str
        The name of the format. This is used on the mediatype to select
        the proper format.
    loads : callable[bytes -> any]
        The function that loads python objects out of the serialized data.
    dumps : callable[any -> bytes]
        The function that serializes python objects to some other format.
    data_loads : callable[any -> any], optional
    data_dumps : callable[any -> any], optional
        Specialized functions for loading and writing only the 'data' field of
        blaze server responses. This allows us to define a more efficient
        serialization format for moving large amounts of data while still
        having a rich representation for the rest of the metadata.
    materialize : callable[(any, DataShape, **kwargs) -> any], optional
        The function used to materialize the result of compute into a form
        suitable for serialization.
    """
    def __init__(self,
                 name,
                 loads,
                 dumps,
                 data_loads=None,
                 data_dumps=None,
                 materialize=None):
        self.name = name
        self.loads = loads
        self.dumps = dumps
        self.data_loads = identity if data_loads is None else data_loads
        self.data_dumps = identity if data_dumps is None else data_dumps
        self.materialize = (
            default_materialize
            if materialize is None else
            materialize
        )

    def __repr__(self):
        return '<%s: %r>' % (type(self).__name__, self.name)
    __str__ = __repr__


def default_materialize(data, dshape, odo_kwargs):
    if iscollection(dshape):
        return odo(data, list, **odo_kwargs)
    if isscalar(dshape):
        return coerce_scalar(data, str(dshape), odo_kwargs)

    return data


def _coerce_str(bytes_or_str):
    if isinstance(bytes_or_str, unicode):
        return bytes_or_str
    return bytes_or_str.decode('utf-8')


json = SerializationFormat(
    'json',
    lambda data: json_module.loads(_coerce_str(data), object_hook=object_hook),
    partial(json_module.dumps, default=json_dumps),
)
pickle = SerializationFormat(
    'pickle',
    pickle_module.loads,
    partial(pickle_module.dumps, protocol=pickle_module.HIGHEST_PROTOCOL),
)

msgpack = SerializationFormat(
    'msgpack',
    partial(msgpack_module.unpackb, encoding='utf-8', object_hook=object_hook),
    partial(msgpack_module.packb, default=json_dumps),
)


try:
    import blosc
    del blosc
    compress = 'blosc'
except ImportError:
    compress = None


def _final_base(values):
    base = values.base
    while True:
        try:
            base = base.base
        except AttributeError:
            return base


def fastmsgpack_object_hook(ob):
    typ = ob.get('typ')
    if typ is None:
        return ob
    if typ == 'nat':
        return pd.NaT
    if typ == 'block_manager':
        df = pd_msgpack_object_hook(ob)
        for block in df._data.blocks:
            values = block.values
            if not values.flags.writeable:
                if sys.getrefcount(_final_base(values)) == 2:
                    # Compatibility shim until this is fixed in pandas
                    # See: https://github.com/pydata/pandas/pull/12359
                    # Mark that we own the memory for this block.
                    # We know that we own it because the only 2 references
                    # for the base object are the values of this block and
                    # TOS because we are passing this object to a function.
                    values.flags.writeable = True
                else:
                    # If we cannot 'safely' make the bytes object mutable
                    # then just copy the data.
                    block.values = values.copy()
        return df
    return pd_msgpack_object_hook(ob)


if PY2:
    def _l1(bs):
        if isinstance(bs, unicode):
            return bs.encode('latin1')
        return bs
else:
    def _l1(bs):
        return bs


def fastmsgpack_data_loads(data):
    raw = list(msgpack_unpack(
        BytesIO(_l1(data)),
        object_hook=fastmsgpack_object_hook,
    ))
    # raw will always be a list, which is most likely a list containing
    # a single dataframe or series
    if len(raw) == 1:
        # we only serialized one structure, just return it
        return raw[0]
    return raw


def fastmsgpack_loads(data):
    raw = list(msgpack_unpack(
        BytesIO(_l1(data)),
        object_hook=object_hook,
    ))
    # raw will always be a list, which is most likely a list containing
    # a single dataframe or series
    if len(raw) == 1:
        # we only serialized one structure, just return it
        return raw[0]
    return raw


def fastmsgpack_default(ob):
    # Compatibility shim until this is fixed in pandas
    # See: https://github.com/pydata/pandas/pull/12307
    if ob is pd.NaT:
        return {'typ': 'nat'}
    return pd_msgpack_default(ob)


def fastmsgpack_data_dumps(data):
    return {
        '__!bytes': pd.to_msgpack(
            None,
            data,
            compress=compress,
            default=fastmsgpack_default,
            encoding='latin1',
        ),
    }


def fastmsgpack_dumps(data):
    return pd.to_msgpack(
        None,
        data,
        compress=compress,
        default=json_dumps,
        encoding='latin1',
    )


def fastmsgpack_materialize(data, dshape, odo_kwargs):
    if istabular(dshape):
        return odo(data, pd.DataFrame, **odo_kwargs)
    if iscollection(dshape):
        return odo(data, pd.Series, **odo_kwargs)
    if isscalar(dshape):
        return coerce_scalar(data, str(dshape), odo_kwargs)
    return data


fastmsgpack = SerializationFormat(
    'fastmsgpack',
    loads=fastmsgpack_loads,
    dumps=fastmsgpack_dumps,
    data_loads=fastmsgpack_data_loads,
    data_dumps=fastmsgpack_data_dumps,
    materialize=fastmsgpack_materialize,
)


all_formats = frozenset(
    g for _, g in globals().items() if isinstance(g, SerializationFormat)
)


__all__ = ['all_formats'] + list(f.name for f in all_formats)
