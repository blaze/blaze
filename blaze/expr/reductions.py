from __future__ import absolute_import, division, print_function

import toolz
from datashape import Record, DataShape, dshape
from datashape import coretypes as ct
import datashape
from numpy import inf

from .core import common_subexpression
from .expressions import Expr, ndim


class Reduction(Expr):
    """ A column-wise reduction

    Blaze supports the same class of reductions as NumPy and Pandas.

        sum, min, max, any, all, mean, var, std, count, nunique

    Examples
    --------

    >>> from blaze import symbol
    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> e = t['amount'].sum()

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data)
    350
    """
    __slots__ = '_hash', '_child', 'axis', 'keepdims'

    def __init__(self, _child, axis=None, keepdims=False):
        self._child = _child
        if axis is None:
            axis = tuple(range(_child.ndim))
        if isinstance(axis, (set, list)):
            axis = tuple(axis)
        if not isinstance(axis, tuple):
            axis = (axis,)
        axis = tuple(sorted(axis))
        self.axis = axis
        self.keepdims = keepdims

    @property
    def dshape(self):
        axis = self.axis
        if self.keepdims:
            shape = tuple(1 if i in axis else d
                          for i, d in enumerate(self._child.shape))
        else:
            shape = tuple(d
                          for i, d in enumerate(self._child.shape)
                          if i not in axis)
        return DataShape(*(shape + (self.schema,)))

    @property
    def schema(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            result = toolz.first(schema.types)
        else:
            result = schema
        return DataShape(result)

    @property
    def symbol(self):
        return type(self).__name__

    @property
    def _name(self):
        try:
            return self._child._name + '_' + type(self).__name__
        except (AttributeError, ValueError, TypeError):
            return type(self).__name__

    def __str__(self):
        kwargs = list()
        if self.keepdims:
            kwargs.append('keepdims=True')
        if self.axis != tuple(range(self._child.ndim)):
            kwargs.append('axis=' + str(self.axis))
        other = sorted(set(self.__slots__[1:]) - set(['_child', 'axis', 'keepdims']))
        for slot in other:
            kwargs.append('%s=%s' % (slot, getattr(self, slot)))
        name = type(self).__name__
        if kwargs:
            return '%s(%s, %s)' % (name, self._child, ', '.join(kwargs))
        else:
            return '%s(%s)' % (name, self._child)


class any(Reduction):
    schema = dshape(ct.bool_)


class all(Reduction):
    schema = dshape(ct.bool_)


class sum(Reduction):
    @property
    def schema(self):
        return DataShape(datashape.maxtype(super(sum, self).schema))


class max(Reduction):
    pass


class min(Reduction):
    pass


class mean(Reduction):
    schema = dshape(ct.real)


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
    __slots__ = '_hash', '_child', 'unbiased', 'axis', 'keepdims'

    schema = dshape(ct.real)

    def __init__(self, child, unbiased=False, *args, **kwargs):
        self.unbiased = unbiased
        super(var, self).__init__(child, *args, **kwargs)


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
    __slots__ = '_hash', '_child', 'unbiased', 'axis', 'keepdims'

    schema = dshape(ct.real)

    def __init__(self, child, unbiased=False, *args, **kwargs):
        self.unbiased = unbiased
        super(std, self).__init__(child, *args, **kwargs)


class count(Reduction):
    """ The number of non-null elements """
    schema = dshape(ct.int32)


class nunique(Reduction):
    schema = dshape(ct.int32)


class nelements(Reduction):
    """Compute the number of elements in a collection, including missing values.

    See Also
    ---------
    blaze.expr.reductions.count: compute the number of non-null elements

    Examples
    --------
    >>> from blaze import symbol
    >>> t = symbol('t', 'var * {name: string, amount: float64}')
    >>> t[t.amount < 1].nelements()
    nelements(t[t.amount < 1])
    """
    schema = dshape(ct.int32)


def nrows(expr):
    return nelements(expr, axis=(0,))


class Summary(Expr):
    """ A collection of named reductions

    Examples
    --------

    >>> from blaze import symbol
    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> expr = summary(number=t.id.nunique(), sum=t.amount.sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 1]]

    >>> from blaze import compute
    >>> compute(expr, data)
    (2, 350)
    """
    __slots__ = '_hash', '_child', 'names', 'values', 'axis', 'keepdims'

    def __init__(self, _child, names, values, axis=None, keepdims=False):
        self._child = _child
        self.names = names
        self.values = values
        self.keepdims = keepdims
        self.axis = axis

    @property
    def dshape(self):
        axis = self.axis
        if self.keepdims:
            shape = tuple(1 if i in axis else d
                          for i, d in enumerate(self._child.shape))
        else:
            shape = tuple(d
                          for i, d in enumerate(self._child.shape)
                          if i not in axis)
        measure = Record(list(zip(self.names,
                                  [v.schema for v in self.values])))
        return DataShape(*(shape + (measure,)))

    def __str__(self):
        s = 'summary('
        s += ', '.join('%s=%s' % (name, str(val))
                         for name, val in zip(self.fields, self.values))
        if self.keepdims:
            s += ', keepdims=True'
        s += ')'
        return s


def summary(keepdims=False, axis=None, **kwargs):
    items = sorted(kwargs.items(), key=toolz.first)
    names = tuple(map(toolz.first, items))
    values = tuple(map(toolz.second, items))
    child = common_subexpression(*values)

    if len(kwargs) == 1 and not iscollection(child.dshape):
        while not iscollection(child.dshape):
            children = [i for i in child._inputs if isinstance(i, Expr)]
            if len(children) == 1:
                child = children[0]
            else:
                child = common_subexpression(*children)

    if axis is None:
        axis = tuple(range(ndim(child)))
    if isinstance(axis, (set, list)):
        axis = tuple(axis)
    if not isinstance(axis, tuple):
        axis = (axis,)
    return Summary(child, names, values, keepdims=keepdims, axis=axis)


summary.__doc__ = Summary.__doc__


def vnorm(expr, ord=None, axis=None, keepdims=False):
    """ Vector norm

    See np.linalg.norm
    """
    if ord is None or ord == 'fro':
        ord = 2
    if ord == inf:
        return max(abs(expr), axis=axis, keepdims=keepdims)
    elif ord == -inf:
        return min(abs(expr), axis=axis, keepdims=keepdims)
    elif ord == 1:
        return sum(abs(expr), axis=axis, keepdims=keepdims)
    elif ord % 2 == 0:
        return sum(expr**ord, axis=axis, keepdims=keepdims)**(1./ord)
    else:
        return sum(abs(expr)**ord, axis=axis, keepdims=keepdims)**(1./ord)


from datashape.predicates import iscollection, isboolean, isnumeric
from .expressions import dshape_method_list, method_properties

dshape_method_list.extend([
    (iscollection, set([count, min, max, nelements])),
    (lambda ds: len(ds.shape) == 1,
        set([nrows, nunique])),
    (lambda ds: iscollection(ds) and isboolean(ds),
        set([any, all, sum])),
    (lambda ds: iscollection(ds) and isnumeric(ds),
        set([mean, sum, mean, min, max, std, var, vnorm])),
    ])

method_properties.update([nrows])
