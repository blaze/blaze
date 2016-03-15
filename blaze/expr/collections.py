from __future__ import absolute_import, division, print_function

import numbers
import numpy as np
from functools import partial
from itertools import chain

import datashape
from datashape import (DataShape, Option, Record, Unit, dshape, var, Fixed,
                       Var, promote, object_)
from datashape.predicates import isscalar, iscollection, isrecord
from toolz import (isdistinct, frequencies, concat as tconcat, unique, get,
                   first, compose, keymap)
import toolz.curried.operator as op
from odo.utils import copydoc

from .core import common_subexpression
from .expressions import Expr, ElemWise, label, Field
from .expressions import dshape_method_list
from ..compatibility import zip_longest, _strtypes
from ..utils import listpack


__all__ = ['Concat',
           'concat',
           'Distinct',
           'distinct',
           'Head',
           'head',
           'IsIn',
           'isin',
           'Join',
           'join',
           'Merge',
           'merge',
           'Sample',
           'sample',
           'Shift',
           'shift',
           'Sort',
           'sort',
           'Tail',
           'tail',
           'transform']


class Sort(Expr):

    """ Table in sorted order

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.sort('amount', ascending=False).schema
    dshape("{name: string, amount: int32}")

    Some backends support sorting by arbitrary rowwise tables, e.g.

    >>> accounts.sort(-accounts.amount) # doctest: +SKIP
    """
    __slots__ = '_hash', '_child', '_key', 'ascending'

    def _dshape(self):
        return self._child.dshape

    @property
    def key(self):
        if self._key is () or self._key is None:
            return self._child.fields[0]
        if isinstance(self._key, tuple):
            return list(self._key)
        else:
            return self._key

    def _len(self):
        return self._child._len()

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return "%s.sort(%s, ascending=%s)" % (self._child, repr(self._key),
                                              self.ascending)


def sort(child, key=None, ascending=True):
    """ Sort a collection

    Parameters
    ----------
    key : str, list of str, or Expr
        Defines by what you want to sort.

          * A single column string: ``t.sort('amount')``
          * A list of column strings: ``t.sort(['name', 'amount'])``
          * An expression: ``t.sort(-t.amount)``

    ascending : bool, optional
        Determines order of the sort
    """
    if not isrecord(child.dshape.measure):
        key = None
    if isinstance(key, list):
        key = tuple(key)
    return Sort(child, key, ascending)


class Distinct(Expr):

    """ Remove duplicate elements from an expression

    Parameters
    ----------
    on : tuple of :class:`~blaze.expr.expressions.Field`
        The subset of fields or names of fields to be distinct on.

    Examples
    --------
    >>> from blaze import symbol
    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> e = distinct(t)

    >>> data = [('Alice', 100, 1),
    ...         ('Bob', 200, 2),
    ...         ('Alice', 100, 1)]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 100, 1), ('Bob', 200, 2)]

    Use a subset by passing `on`:

    >>> import pandas as pd
    >>> e = distinct(t, 'name')
    >>> data = pd.DataFrame([['Alice', 100, 1],
    ...                      ['Alice', 200, 2],
    ...                      ['Bob', 100, 1],
    ...                      ['Bob', 200, 2]],
    ...                     columns=['name', 'amount', 'id'])
    >>> compute(e, data)
        name  amount  id
    0  Alice     100   1
    1    Bob     100   1


    """
    __slots__ = '_hash', '_child', 'on'

    def _dshape(self):
        return datashape.var * self._child.dshape.measure

    @property
    def fields(self):
        return self._child.fields

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return 'distinct({child}{on})'.format(
            child=self._child,
            on=(', ' if self.on else '') + ', '.join(map(str, self.on))
        )


@copydoc(Distinct)
def distinct(expr, *on):
    fields = frozenset(expr.fields)
    _on = []
    append = _on.append
    for n in on:
        if isinstance(n, Field):
            if n._child.isidentical(expr):
                n = n._name
            else:
                raise ValueError('{0} is not a field of {1}'.format(n, expr))
        if not isinstance(n, _strtypes):
            raise TypeError('on must be a name or field, not: {0}'.format(n))
        elif n not in fields:
            raise ValueError('{0} is not a field of {1}'.format(n, expr))
        append(n)

    return Distinct(expr, tuple(_on))


class _HeadOrTail(Expr):
    __slots__ = '_hash', '_child', 'n'

    def _dshape(self):
        return self.n * self._child.dshape.subshape[0]

    def _len(self):
        return min(self._child._len(), self.n)

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return '%s.%s(%d)' % (self._child, type(self).__name__.lower(), self.n)


class Head(_HeadOrTail):

    """ First `n` elements of collection

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.head(5).dshape
    dshape("5 * {name: string, amount: int32}")

    See Also
    --------

    blaze.expr.collections.Tail
    """
    pass


@copydoc(Head)
def head(child, n=10):
    return Head(child, n)


class Tail(_HeadOrTail):
    """ Last `n` elements of collection

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.tail(5).dshape
    dshape("5 * {name: string, amount: int32}")

    See Also
    --------

    blaze.expr.collections.Head
    """
    pass


@copydoc(Tail)
def tail(child, n=10):
    return Tail(child, n)


class Sample(Expr):
    """Random row-wise sample.  Can specify `n` or `frac` for an absolute or
    fractional number of rows, respectively.

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.sample(n=2).dshape
    dshape("var * {name: string, amount: int32}")
    >>> accounts.sample(frac=0.1).dshape
    dshape("var * {name: string, amount: int32}")
    """
    __slots__ = '_hash', '_child', 'n', 'frac'

    def _dshape(self):
        return self._child.dshape

    def __str__(self):
        arg = 'n={}'.format(self.n) if self.n is not None else 'frac={}'.format(self.frac)
        return '%s.sample(%s)' % (self._child, arg)


@copydoc(Sample)
def sample(child, n=None, frac=None):
    if n is frac is None:
        raise TypeError("sample() missing 1 required argument, 'n' or 'frac'.")
    if n is not None and frac is not None:
        raise ValueError("n ({}) and frac ({}) cannot both be specified.".format(n, frac))
    if n is not None:
        n = op.index(n)
        if n < 1:
            raise ValueError("n must be positive, given {}".format(n))
    if frac is not None:
        frac = float(frac)
        if not 0.0 <= frac <= 1.0:
            raise ValueError("sample requires 0 <= frac <= 1.0, given {}".format(frac))
    return Sample(child, n, frac)


def transform(t, replace=True, **kwargs):
    """ Add named columns to table

    >>> from blaze import symbol
    >>> t = symbol('t', 'var * {x: int, y: int}')
    >>> transform(t, z=t.x + t.y).fields
    ['x', 'y', 'z']
    """
    if replace and set(t.fields).intersection(set(kwargs)):
        t = t[[c for c in t.fields if c not in kwargs]]

    args = [t] + [v.label(k) for k, v in sorted(kwargs.items(), key=first)]
    return merge(*args)


def schema_concat(exprs):
    """ Concatenate schemas together.  Supporting both Records and Units

    In the case of Units, the name is taken from expr.name
    """
    new_fields = []
    for c in exprs:
        schema = c.schema[0]
        if isinstance(schema, Record):
            new_fields.extend(schema.fields)
        elif isinstance(schema, (Unit, Option)):
            new_fields.append((c._name, schema))
        else:
            raise TypeError("All schemas must have Record or Unit shape."
                            "\nGot %s" % schema)
    return dshape(Record(new_fields))


class Merge(ElemWise):

    """ Merge many fields together

    Examples
    --------
    >>> from blaze import symbol, label
    >>> accounts = symbol('accounts', 'var * {name: string, x: int, y: real}')
    >>> merge(accounts.name, z=accounts.x + accounts.y).fields
    ['name', 'z']

    To control the ordering of the fields, use ``label``:

    >>> merge(label(accounts.name, 'NAME'), label(accounts.x, 'X')).dshape
    dshape("var * {NAME: string, X: int32}")
    >>> merge(label(accounts.x, 'X'), label(accounts.name, 'NAME')).dshape
    dshape("var * {X: int32, NAME: string}")
    """
    __slots__ = '_hash', '_child', 'children'

    def _schema(self):
        return schema_concat(self.children)

    @property
    def fields(self):
        return list(tconcat(child.fields for child in self.children))

    def _subterms(self):
        yield self
        for i in self.children:
            for node in i._subterms():
                yield node

    def _get_field(self, key):
        for child in self.children:
            if key in child.fields:
                if isscalar(child.dshape.measure):
                    return child
                else:
                    return child[key]

    def _project(self, key):
        if not isinstance(key, (tuple, list)):
            raise TypeError("Expected tuple or list, got %s" % key)
        return merge(*[self[c] for c in key])

    def _leaves(self):
        return list(unique(tconcat(i._leaves() for i in self.children)))


@copydoc(Merge)
def merge(*exprs, **kwargs):
    if len(exprs) + len(kwargs) == 1:
        if exprs:
            return exprs[0]
        if kwargs:
            [(k, v)] = kwargs.items()
            return v.label(k)
    # Get common sub expression
    exprs += tuple(label(v, k) for k, v in sorted(kwargs.items(), key=first))
    child = common_subexpression(*exprs)
    result = Merge(child, exprs)

    if not isdistinct(result.fields):
        raise ValueError(
            "Repeated columns found: " + ', '.join(
                k for k, v in frequencies(result.fields).items() if v > 1
            ),
        )

    return result


def unpack(l):
    """ Unpack items from collections of nelements 1

    >>> unpack('hello')
    'hello'
    >>> unpack(['hello'])
    'hello'
    """
    if isinstance(l, (tuple, list, set)) and len(l) == 1:
        return next(iter(l))
    else:
        return l


class Join(Expr):

    """ Join two tables on common columns

    Parameters
    ----------
    lhs, rhs : Expr
        Expressions to join
    on_left : str, optional
        The fields from the left side to join on.
        If no ``on_right`` is passed, then these are the fields for both
        sides.
    on_right : str, optional
        The fields from the right side to join on.
    how : {'inner', 'outer', 'left', 'right'}
        What type of join to perform.
    suffixes: pair of str
        The suffixes to be applied to the left and right sides
        in order to resolve duplicate field names.

    Examples
    --------
    >>> from blaze import symbol
    >>> names = symbol('names', 'var * {name: string, id: int}')
    >>> amounts = symbol('amounts', 'var * {amount: int, id: int}')

    Join tables based on shared column name

    >>> joined = join(names, amounts, 'id')

    Join based on different column names

    >>> amounts = symbol('amounts', 'var * {amount: int, acctNumber: int}')
    >>> joined = join(names, amounts, 'id', 'acctNumber')

    See Also
    --------

    blaze.expr.collections.Merge
    """
    __slots__ = (
        '_hash', 'lhs', 'rhs', '_on_left', '_on_right', 'how', 'suffixes'
    )
    __inputs__ = 'lhs', 'rhs'

    @property
    def on_left(self):
        on_left = self._on_left
        if isinstance(on_left, tuple):
            return list(on_left)
        return on_left

    @property
    def on_right(self):
        on_right = self._on_right
        if isinstance(on_right, tuple):
            return list(on_right)
        return on_right

    def _schema(self):
        """

        Examples
        --------
        >>> from blaze import symbol
        >>> t = symbol('t', 'var * {name: string, amount: int}')
        >>> s = symbol('t', 'var * {name: string, id: int}')

        >>> join(t, s).schema
        dshape("{name: string, amount: int32, id: int32}")

        >>> join(t, s, how='left').schema
        dshape("{name: string, amount: int32, id: ?int32}")

        Overlapping but non-joined fields append _left, _right

        >>> a = symbol('a', 'var * {x: int, y: int}')
        >>> b = symbol('b', 'var * {x: int, y: int}')
        >>> join(a, b, 'x').fields
        ['x', 'y_left', 'y_right']
        """

        option = lambda dt: dt if isinstance(dt, Option) else Option(dt)

        on_left = self.on_left
        if not isinstance(on_left, list):
            on_left = on_left,

        on_right = self.on_right
        if not isinstance(on_right, list):
            on_right = on_right,

        right_types = keymap(
            dict(zip(on_right, on_left)).get,
            self.rhs.dshape.measure.dict,
        )
        joined = (
            (name, promote(dt, right_types[name], promote_option=False))
            for n, (name, dt) in enumerate(filter(
                compose(op.contains(on_left), first),
                self.lhs.dshape.measure.fields,
            ))
        )

        left = [
            (name, dt) for name, dt in zip(
                self.lhs.fields,
                types_of_fields(self.lhs.fields, self.lhs)
            ) if name not in on_left
        ]

        right = [
            (name, dt) for name, dt in zip(
                self.rhs.fields,
                types_of_fields(self.rhs.fields, self.rhs)
            ) if name not in on_right
        ]

        # Handle overlapping but non-joined case, e.g.
        left_other = set(name for name, dt in left if name not in on_left)
        right_other = set(name for name, dt in right if name not in on_right)
        overlap = left_other & right_other

        left_suffix, right_suffix = self.suffixes
        left = ((name + left_suffix if name in overlap else name, dt)
                for name, dt in left)
        right = ((name + right_suffix if name in overlap else name, dt)
                 for name, dt in right)

        if self.how in ('right', 'outer'):
            left = ((name, option(dt)) for name, dt in left)
        if self.how in ('left', 'outer'):
            right = ((name, option(dt)) for name, dt in right)

        return dshape(Record(chain(joined, left, right)))

    def _dshape(self):
        # TODO: think if this can be generalized
        return var * self.schema


def types_of_fields(fields, expr):
    """ Get the types of fields in an expression

    Examples
    --------
    >>> from blaze import symbol
    >>> expr = symbol('e', 'var * {x: int64, y: float32}')
    >>> types_of_fields('y', expr)
    ctype("float32")

    >>> types_of_fields(['y', 'x'], expr)
    (ctype("float32"), ctype("int64"))

    >>> types_of_fields('x', expr.x)
    ctype("int64")
    """
    if isinstance(expr.dshape.measure, Record):
        return get(fields, expr.dshape.measure)
    else:
        if isinstance(fields, (tuple, list, set)):
            assert len(fields) == 1
            fields, = fields
        assert fields == expr._name
        return expr.dshape.measure


@copydoc(Join)
def join(lhs, rhs, on_left=None, on_right=None,
         how='inner', suffixes=('_left', '_right')):
    if not on_left and not on_right:
        on_left = on_right = unpack(list(sorted(
            set(lhs.fields) & set(rhs.fields),
            key=lhs.fields.index)))
    if not on_right:
        on_right = on_left
    if isinstance(on_left, tuple):
        on_left = list(on_left)
    if isinstance(on_right, tuple):
        on_right = list(on_right)
    if not on_left or not on_right:
        raise ValueError(
            "Can not Join.  No shared columns between %s and %s" % (lhs, rhs),
        )
    left_types = listpack(types_of_fields(on_left, lhs))
    right_types = listpack(types_of_fields(on_right, rhs))
    if len(left_types) != len(right_types):
        raise ValueError(
            'Length of on_left=%d not equal to length of on_right=%d' % (
                len(left_types), len(right_types),
            ),
        )

    for n, promotion in enumerate(map(partial(promote, promote_option=False),
                                      left_types,
                                      right_types)):
        if promotion == object_:
            raise TypeError(
                'Schemata of joining columns do not match,'
                ' no promotion found for %s=%s and %s=%s' % (
                    on_left[n], left_types[n], on_right[n], right_types[n],
                ),
            )
    _on_left = tuple(on_left) if isinstance(on_left, list) else on_left
    _on_right = (tuple(on_right) if isinstance(on_right, list)
                 else on_right)

    how = how.lower()
    if how not in ('inner', 'outer', 'left', 'right'):
        raise ValueError("How parameter should be one of "
                         "\n\tinner, outer, left, right."
                         "\nGot: %s" % how)

    return Join(lhs, rhs, _on_left, _on_right, how, suffixes)


class Concat(Expr):

    """ Stack tables on common columns

    Parameters
    ----------
    lhs, rhs : Expr
        Collections to concatenate
    axis : int, optional
        The axis to concatenate on.

    Examples
    --------
    >>> from blaze import symbol

    Vertically stack tables:

    >>> names = symbol('names', '5 * {name: string, id: int32}')
    >>> more_names = symbol('more_names', '7 * {name: string, id: int32}')
    >>> stacked = concat(names, more_names)
    >>> stacked.dshape
    dshape("12 * {name: string, id: int32}")

    Vertically stack matrices:

    >>> mat_a = symbol('a', '3 * 5 * int32')
    >>> mat_b = symbol('b', '3 * 5 * int32')
    >>> vstacked = concat(mat_a, mat_b, axis=0)
    >>> vstacked.dshape
    dshape("6 * 5 * int32")

    Horizontally stack matrices:

    >>> hstacked = concat(mat_a, mat_b, axis=1)
    >>> hstacked.dshape
    dshape("3 * 10 * int32")

    See Also
    --------

    blaze.expr.collections.Merge
    """
    __slots__ = '_hash', 'lhs', 'rhs', 'axis'
    __inputs__ = 'lhs', 'rhs'

    def _dshape(self):
        axis = self.axis
        ldshape = self.lhs.dshape
        lshape = ldshape.shape
        return DataShape(
            *(lshape[:axis] + (
                _shape_add(lshape[axis], self.rhs.dshape.shape[axis]),
            ) + lshape[axis + 1:] + (ldshape.measure,))
        )


def _shape_add(a, b):
    if isinstance(a, Var) or isinstance(b, Var):
        return var
    return Fixed(a.val + b.val)


@copydoc(Concat)
def concat(lhs, rhs, axis=0):
    ldshape = lhs.dshape
    rdshape = rhs.dshape
    if ldshape.measure != rdshape.measure:
        raise TypeError(
            'Mismatched measures: {l} != {r}'.format(
                l=ldshape.measure, r=rdshape.measure
            ),
        )

    lshape = ldshape.shape
    rshape = rdshape.shape
    for n, (a, b) in enumerate(zip_longest(lshape, rshape, fillvalue=None)):
        if n != axis and a != b:
            raise TypeError(
                'Shapes are not equal along axis {n}: {a} != {b}'.format(
                    n=n, a=a, b=b,
                ),
            )
    if axis < 0 or 0 < len(lshape) <= axis:
        raise ValueError(
            "Invalid axis '{a}', must be in range: [0, {n})".format(
                a=axis, n=len(lshape)
            ),
        )

    return Concat(lhs, rhs, axis)


class IsIn(ElemWise):
    """Check if an expression contains values from a set.

    Return a boolean expression indicating whether another expression
    contains values that are members of a collection.

    Parameters
    ----------
    expr : Expr
        Expression whose elements to check for membership in `keys`
    keys : Sequence
        Elements to test against. Blaze stores this as a ``frozenset``.

    Examples
    --------

    Check if a vector contains any of 1, 2 or 3:

    >>> from blaze import symbol
    >>> t = symbol('t', '10 * int64')
    >>> expr = t.isin([1, 2, 3])
    >>> expr.dshape
    dshape("10 * bool")
    """
    __slots__ = '_hash', '_child', '_keys'

    def _schema(self):
        return datashape.bool_

    def __str__(self):
        return '%s.%s(%s)' % (self._child, type(self).__name__.lower(),
                              self._keys)


@copydoc(IsIn)
def isin(expr, keys):
    if isinstance(keys, Expr):
        raise TypeError('keys argument cannot be an expression, '
                        'it must be an iterable object such as a list, '
                        'tuple or set')
    return IsIn(expr, frozenset(keys))


class Shift(Expr):
    """ Shift a column backward or forward by N elements

    Parameters
    ----------
    expr : Expr
        The expression to shift. This expression's dshape should be columnar
    n : int
        The number of elements to shift by. If n < 0 then shift backward,
        if n == 0 do nothing, else shift forward.
    """
    __slots__ = '_hash', '_child', 'n'

    def _schema(self):
        measure = self._child.schema.measure

        # if we are not shifting or we are already an Option type then return
        # the child's schema
        if not self.n or isinstance(measure, Option):
            return measure
        else:
            return Option(measure)

    def _dshape(self):
        return DataShape(*(self._child.dshape.shape + tuple(self.schema)))

    def __str__(self):
        return '%s(%s, n=%d)' % (
            type(self).__name__.lower(), self._child, self.n
        )


@copydoc(Shift)
def shift(expr, n):
    if not isinstance(n, (numbers.Integral, np.integer)):
        raise TypeError('n must be an integer')
    return Shift(expr, n)


dshape_method_list.extend([(iscollection, set([sort, head, tail, sample])),
                           (lambda ds: len(ds.shape) == 1, set([distinct, shift])),
                           (lambda ds: (len(ds.shape) == 1 and
                                        isscalar(getattr(ds.measure, 'key', ds.measure))), set([isin])),
                           (lambda ds: len(ds.shape) == 1 and isscalar(ds.measure), set([isin]))])
