from __future__ import absolute_import, division, print_function

import itertools

import datashape
from datashape import discover, Tuple, Record, DataShape, var
from datashape.predicates import (
    istabular,
    isscalar,
    isrecord,
)
from odo import resource
from odo.utils import ignoring, copydoc

from ..compatibility import collections_abc, _strtypes
from ..dispatch import dispatch
from .expressions import sanitized_dshape, Symbol

__all__ = ['BoundSymbol', 'Literal', 'Data', 'literal', 'data']


_names = ('_%d' % i for i in itertools.count(1))
not_an_iterator = []


with ignoring(ImportError):
    import bcolz
    not_an_iterator.append(bcolz.carray)


with ignoring(ImportError):
    import pymongo
    not_an_iterator.append(pymongo.collection.Collection)
    not_an_iterator.append(pymongo.database.Database)


class generate(object):
    """A sentinel value, indicating whether or not `literal` should
    generate a name for the returned `BoundSymbol`.
    """

    def __new__(cls):
        raise NotImplementedError('Can not create instance of sentinel type.')


class BoundSymbol(Symbol):

    # NOTE: This docstring is meant to correspond to the ``literal()`` and
    # ``data()`` APIs, which is why the Parameters section doesn't match the
    # arguments to ``Literal.__new__()``.

    """Bind a data resource to a symbol, for use in expressions and
    computation.

    A ``data`` object presents a consistent view onto a variety of concrete
    data sources.  Like ``symbol`` objects, they are meant to be used in
    expressions.  Because they are tied to concrete data resources, ``data``
    objects can be used with ``compute`` directly, making them convenient for
    interactive exploration.

    Parameters
    ----------
    data_source : object
        Any type with ``discover`` and ``compute`` implementations
    fields : list, optional
        Field or column names, will be inferred from data_source if possible
    dshape : str or DataShape, optional
        DataShape describing input data
    name : str, optional
        A name for the data.

    Examples
    --------
    >>> t = data([(1, 'Alice', 100),
    ...           (2, 'Bob', -200),
    ...           (3, 'Charlie', 300),
    ...           (4, 'Denis', 400),
    ...           (5, 'Edith', -500)],
    ...          fields=['id', 'name', 'balance'])
    >>> t[t.balance < 0].name.peek()
        name
    0    Bob
    1  Edith
    """
    _arguments = 'data', 'dshape', '_name'

    def _resources(self):
        return {self: self.data}

    @property
    def _token(self):
        return 0

    @classmethod
    def _static_identity(cls, data, dshape, _name):
        try:
            # cannot use isinstance(data, Hashable)
            # some classes give a false positive
            hash(data)
        except TypeError:
            data = id(data)
        return cls, data, dshape, _name

    def __str__(self):
        name = self._name
        return name if name is not None else repr(self)


class Literal(BoundSymbol):
    def __repr__(self):
        name = self._name
        return name if name is not None else repr(self.data)


class Data(BoundSymbol):
    def __repr__(self):
        return "<'{}' data; _name='{}', dshape='{}'>".format(
            type(self.data).__name__,
            self._name,
            sanitized_dshape(self.dshape),
        )


def _bound_symbol(cls,
                  data_source,
                  dshape,
                  name,
                  fields,
                  schema,
                  **kwargs):
    if schema and dshape:
        raise ValueError(
            'Please specify one of schema= or dshape= keyword  arguments',
        )

    if isinstance(data_source, BoundSymbol):
        return _bound_symbol(
            cls,
            data_source.data,
            dshape,
            name,
            fields,
            schema,
            **kwargs
        )

    if schema and not dshape:
        dshape = var * schema
    if dshape and isinstance(dshape, _strtypes):
        dshape = datashape.dshape(dshape)

    if isinstance(data_source, _strtypes):
        data_source = resource(
            data_source,
            schema=schema,
            dshape=dshape,
            **kwargs
        )

    if (isinstance(data_source, collections_abc.Iterator) and
            not isinstance(data_source, tuple(not_an_iterator))):
        data_source = tuple(data_source)

    if not dshape:
        dshape = discover(data_source)
        types = None
        if isinstance(dshape.measure, Tuple) and fields:
            types = dshape[1].dshapes
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape + (schema,)))
        elif isscalar(dshape.measure) and fields:
            types = (dshape.measure,) * int(dshape[-2])
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape[:-1] + (schema,)))
        elif isrecord(dshape.measure) and fields:
            ds = discover(data_source)
            assert isrecord(ds.measure)
            names = ds.measure.names
            if names != fields:
                raise ValueError(
                    'data column names %s\n'
                    '\tnot equal to fields parameter %s,\n'
                    '\tuse data(data_source).relabel(%s) to rename '
                    'fields' % (
                        names,
                        fields,
                        ', '.join(
                            '%s=%r' % (k, v)
                            for k, v in
                            zip(names, fields)
                        ),
                    ),
                )
            types = dshape.measure.types
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape + (schema,)))

    ds = datashape.dshape(dshape)

    if name is generate:
        if istabular(dshape):
            name = next(_names)
        else:
            name = None

    return cls(data_source, ds, name)


@copydoc(BoundSymbol)
def literal(data_source,
            dshape=None,
            name=None,
            fields=None,
            schema=None,
            **kwargs):
    return _bound_symbol(
        Literal,
        data_source,
        dshape=dshape,
        name=name,
        fields=fields,
        schema=schema,
        **kwargs
    )


@copydoc(BoundSymbol)
def data(data_source,
         dshape=None,
         name=generate,
         fields=None,
         schema=None,
         **kwargs):
    return _bound_symbol(
        Data,
        data_source,
        dshape=dshape,
        name=name,
        fields=fields,
        schema=schema,
        **kwargs
    )


@dispatch(BoundSymbol, collections_abc.Mapping)
def _subs(o, d):
    return o
