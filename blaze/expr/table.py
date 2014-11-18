""" An abstract Table

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts.name[accounts.amount < 0]
"""
from __future__ import absolute_import, division, print_function

import datashape
from datashape.predicates import isscalar, iscollection, isrecord
from .expressions import Symbol
from ..compatibility import _strtypes

__all__ = ['TableSymbol']


def TableSymbol(name, dshape):
    """ A Symbol for Tabular data

    This is a leaf in the expression tree

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts.amount + 1
    accounts.amount + 1

    We define a TableSymbol with a name like ``accounts`` and the datashape of
    a single row, called a schema.
    """
    if isinstance(dshape, _strtypes):
        dshape = datashape.dshape(dshape)
    if not iscollection(dshape):
        dshape = datashape.var * dshape
    return Symbol(name, dshape)


def columns(expr):
    return expr.fields


from .expressions import dshape_method_list, method_properties

dshape_method_list.extend([
    (lambda ds: len(ds.shape) == 1 and isrecord(ds.measure), set([columns]))
    ])

method_properties.add(columns)
