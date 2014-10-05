from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import functools
import datashape
from datashape import dshape, DataShape, Record, Var, Option, Unit

from toolz import unique, concat, memoize, partial, first
import toolz
from blaze.compatibility import _strtypes, builtins

from .core import *
from .arithmetic import (Eq, Ne, Lt, Le, Gt, Ge, Add, Mult, Div, Sub, Pow, Mod, Or,
                     And, USub, Not, FloorDiv)
from datashape.predicates import isscalar, iscollection
from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Collection', 'Projection', 'projection', 'Selection', 'selection',
'Label', 'label', 'ElemWise']


class Collection(Expr):
    pass



class Selection(Collection):
    """ Filter elements of expression based on predicate

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> deadbeats = accounts[accounts['amount'] < 0]
    """
    __slots__ = 'child', 'predicate'

    def __str__(self):
        return "%s[%s]" % (self.child, self.predicate)

    @property
    def dshape(self):
        shape = list(self.child.dshape.shape)
        shape[0] = Var()
        return DataShape(*(shape + [self.child.dshape.measure]))


def selection(table, predicate):
    subexpr = common_subexpression(table, predicate)

    if not builtins.all(isinstance(node, (ElemWise, Symbol))
                        or node.isidentical(subexpr)
           for node in concat([path(predicate, subexpr),
                               path(table, subexpr)])):

        raise ValueError("Selection not properly matched with table:\n"
                   "child: %s\n"
                   "apply: %s\n"
                   "predicate: %s" % (subexpr, table, predicate))

    if predicate.dtype != dshape('bool'):
        raise TypeError("Must select over a boolean predicate.  Got:\n"
                        "%s[%s]" % (table, predicate))

    return table.subs({subexpr: Selection(subexpr, predicate)})

selection.__doc__ = Selection.__doc__


class Label(ElemWise):
    """ A Labeled expresion

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> (accounts['amount'] * 100)._name
    'amount'

    >>> (accounts['amount'] * 100).label('new_amount')._name
    'new_amount'

    See Also
    --------

    blaze.expr.table.ReLabel
    """
    __slots__ = 'child', 'label'

    @property
    def schema(self):
        return self.child.schema

    @property
    def _name(self):
        return self.label

    def get_field(self, key):
        if key[0] == self.fields[0]:
            return self
        else:
            raise ValueError("Column Mismatch: %s" % key)

def label(expr, lab):
    return Label(expr, lab)
label.__doc__ = Label.__doc__


class Map(ElemWise):
    """ Map an arbitrary Python function across elements in a collection

    Examples
    --------

    >>> from datetime import datetime
    >>> from blaze import TableSymbol

    >>> t = TableSymbol('t', '{price: real, time: int64}')  # times as integers
    >>> datetimes = t['time'].map(datetime.utcfromtimestamp)

    Optionally provide extra schema information

    >>> datetimes = t['time'].map(datetime.utcfromtimestamp,
    ...                           schema='{time: datetime}')

    See Also
    --------

    blaze.expr.table.Apply
    """
    __slots__ = 'child', 'func', '_schema', '_name0'

    @property
    def schema(self):
        if self._schema:
            return dshape(self._schema)
        else:
            raise NotImplementedError("Schema of mapped column not known.\n"
                    "Please specify schema keyword in .map method.\n"
                    "t['columnname'].map(function, schema='{col: type}')")

    def label(self, name):
        assert isscalar(self.dshape.measure)
        return Map(self.child,
                   self.func,
                   self.schema,
                   name)

    @property
    def shape(self):
        return self.child.shape

    @property
    def ndim(self):
        return self.child.ndim

    @property
    def _name(self):
        if self._name0:
            return self._name0
        else:
            return self.child._name


def shape(expr):
    """ Shape of expression

    >>> Symbol('s', '3 * 5 * int32').shape
    (3, 5)
    """
    s = list(expr.dshape.shape)
    for i, elem in enumerate(s):
        try:
            s[i] = int(elem)
        except TypeError:
            pass

    return tuple(s)


def ndim(expr):
    """ Number of dimensions of expression

    >>> Symbol('s', '3 * var * int32').ndim
    2
    """
    return len(expr.shape)


from .core import dshape_method_list, schema_method_list, method_properties


dshape_method_list.extend([
    (iscollection, {shape, ndim}),
    ])

method_properties.update([shape, ndim])
