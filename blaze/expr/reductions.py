from __future__ import absolute_import, division, print_function

import toolz
from toolz import first
import datashape
from datashape import Record, dshape, DataShape
from datashape import coretypes as ct
from datashape.predicates import isscalar, iscollection

from .core import common_subexpression
from .expressions import Expr, Symbol

class Reduction(Expr):
    """ A column-wise reduction

    Blaze supports the same class of reductions as NumPy and Pandas.

        sum, min, max, any, all, mean, var, std, count, nunique

    Examples
    --------

    >>> t = Symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> e = t['amount'].sum()

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data)
    350
    """
    __slots__ = '_child', 'axis'
    _dtype = None

    def __init__(self, _child, axis=None):
        self._child = _child
        self.axis = axis

    @property
    def dshape(self):
        axis = self.axis
        if axis == None:
            return dshape(self._dtype)
        if isinstance(axis, int):
            axis = [axis]
        s = tuple(slice(None) if i not in axis else 0
                              for i in range(self._child.ndim))
        ds = self._child.dshape.subshape[s]
        return DataShape(*(ds.shape + (self._dtype,)))

    @property
    def symbol(self):
        return type(self).__name__

    @property
    def _name(self):
        try:
            return self._child._name + '_' + type(self).__name__
        except (AttributeError, ValueError, TypeError):
            return type(self).__name__


class any(Reduction):
    _dtype = ct.bool_

class all(Reduction):
    _dtype = ct.bool_

class sum(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class max(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class min(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class mean(Reduction):
    _dtype = ct.real

class var(Reduction):
    """Variance

    Parameters
    ----------
    child : Expr
        An expression
    unbiased : bool, optional
        Compute an unbiased estimate of the population variance if this is
        ``True``. In NumPy and pandas, this parameter is called ``ddof`` (delta
        degrees of freedom) and is equal to 1 for unbiased and 0 for biased.
    """
    __slots__ = '_child', 'unbiased'

    _dtype = ct.real

    def __init__(self, child, unbiased=False):
        super(var, self).__init__(child, unbiased)

class std(Reduction):
    """Standard Deviation

    Parameters
    ----------
    child : Expr
        An expression
    unbiased : bool, optional
        Compute the square root of an unbiased estimate of the population
        variance if this is ``True``.

        .. warning::

            This does *not* return an unbiased estimate of the population
            standard deviation.

    See Also
    --------
    var
    """
    __slots__ = '_child', 'unbiased'

    _dtype = ct.real

    def __init__(self, child, unbiased=False):
        super(std, self).__init__(child, unbiased)

class count(Reduction):
    _dtype = ct.int_

class nunique(Reduction):
    _dtype = ct.int_


class Summary(Expr):
    """ A collection of named reductions

    Examples
    --------

    >>> t = Symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> expr = summary(number=t.id.nunique(), sum=t.amount.sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 1]]

    >>> from blaze.compute.python import compute
    >>> compute(expr, data)
    (2, 350)
    """
    __slots__ = '_child', 'names', 'values'

    @property
    def dshape(self):
        return dshape(Record(list(zip(self.names,
                                      [v._dtype for v in self.values]))))

    def __str__(self):
        return 'summary(' + ', '.join('%s=%s' % (name, str(val))
                for name, val in zip(self.fields, self.values)) + ')'


def summary(**kwargs):
    items = sorted(kwargs.items(), key=first)
    names = tuple(map(first, items))
    values = tuple(map(toolz.second, items))
    child = common_subexpression(*values)

    if len(kwargs) == 1 and not iscollection(child.dshape):
        while not iscollection(child.dshape):
            children = [i for i in child._inputs if isinstance(i, Expr)]
            if len(children) == 1:
                child = children[0]
            else:
                raise ValueError()

    return Summary(child, names, values)


summary.__doc__ = Summary.__doc__


from datashape.predicates import (iscollection, isscalar, isrecord, isboolean,
        isnumeric)
from .expressions import schema_method_list, dshape_method_list

schema_method_list.extend([
    (isboolean, set([any, all, sum])),
    (isnumeric, set([mean, sum, mean, min, max, std, var])),
    ])

dshape_method_list.extend([
    (iscollection, set([count, nunique, min, max])),
    ])
