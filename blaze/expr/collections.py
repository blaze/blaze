from __future__ import absolute_import, division, print_function

from toolz import (
    isdistinct, frequencies, concat as tconcat, unique, get, first,
)
import datashape
from datashape import DataShape, Option, Record, Unit, dshape, var, Fixed, Var
from datashape.predicates import isscalar, iscollection, isrecord

from .core import common_subexpression
from .expressions import Expr, ElemWise, label
from .expressions import dshape_method_list
from ..compatibility import zip_longest


__all__ = ['Sort', 'Distinct', 'Head', 'Merge', 'IsIn', 'isin', 'distinct',
           'merge', 'head', 'sort', 'Join', 'join', 'transform', 'Concat',
           'concat']


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

    @property
    def dshape(self):
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
    """
    __slots__ = '_hash', '_child',

    @property
    def dshape(self):
        return datashape.var * self._child.dshape.measure

    @property
    def fields(self):
        return self._child.fields

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return 'distinct(%s)' % self._child


def distinct(expr):
    return Distinct(expr)


distinct.__doc__ = Distinct.__doc__


class Head(Expr):

    """ First `n` elements of collection

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.head(5).dshape
    dshape("5 * {name: string, amount: int32}")
    """
    __slots__ = '_hash', '_child', 'n'

    @property
    def dshape(self):
        return self.n * self._child.dshape.subshape[0]

    def _len(self):
        return min(self._child._len(), self.n)

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return '%s.head(%d)' % (self._child, self.n)


def head(child, n=10):
    return Head(child, n)

head.__doc__ = Head.__doc__


def merge(*exprs, **kwargs):
    if len(exprs) + len(kwargs) == 1:
        if exprs:
            return exprs[0]
        if kwargs:
            [(k, v)] = kwargs.items()
            return v.label(k)
    # Get common sub expression
    exprs += tuple(label(v, k) for k, v in sorted(kwargs.items(), key=first))
    try:
        child = common_subexpression(*exprs)
    except Exception:
        raise ValueError("No common subexpression found for input expressions")

    result = Merge(child, exprs)

    if not isdistinct(result.fields):
        raise ValueError("Repeated columns found: " + ', '.join(k for k, v in
                                                                frequencies(result.fields).items() if v > 1))

    return result


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
    names, values = [], []
    for c in exprs:
        schema = c.schema[0]
        if isinstance(schema, Option):
            schema = schema.ty
        if isinstance(schema, Record):
            names.extend(schema.names)
            values.extend(schema.types)
        elif isinstance(schema, Unit):
            names.append(c._name)
            values.append(schema)
        else:
            raise TypeError("All schemas must have Record or Unit shape."
                            "\nGot %s" % c.schema[0])
    return dshape(Record(list(zip(names, values))))


class Merge(ElemWise):

    """ Merge many fields together

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, x: int, y: real}')
    >>> merge(accounts.name, z=accounts.x + accounts.y).fields
    ['name', 'z']
    """
    __slots__ = '_hash', '_child', 'children'

    @property
    def schema(self):
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


merge.__doc__ = Merge.__doc__


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
    on_left : string
    on_right : string
    suffixes: pair of strings

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
        '_hash', 'lhs', 'rhs', '_on_left', '_on_right', 'how', 'suffixes',
    )
    __inputs__ = 'lhs', 'rhs'

    @property
    def on_left(self):
        if isinstance(self._on_left, tuple):
            return list(self._on_left)
        else:
            return self._on_left

    @property
    def on_right(self):
        if isinstance(self._on_right, tuple):
            return list(self._on_right)
        else:
            return self._on_right

    @property
    def schema(self):
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

        joined = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                  if name in self.on_left]

        left = [[name, dt] for name, dt in
                zip(self.lhs.fields, types_of_fields(
                    self.lhs.fields, self.lhs))
                if name not in self.on_left]

        right = [[name, dt] for name, dt in
                 zip(self.rhs.fields, types_of_fields(
                     self.rhs.fields, self.rhs))
                 if name not in self.on_right]

        # Handle overlapping but non-joined case, e.g.
        left_other = [name for name, dt in left if name not in self.on_left]
        right_other = [name for name, dt in right if name not in self.on_right]
        overlap = set.intersection(set(left_other), set(right_other))
        left_suffix, right_suffix = self.suffixes
        left = [[name + left_suffix if name in overlap else name, dt]
                for name, dt in left]
        right = [[name + right_suffix if name in overlap else name, dt]
                 for name, dt in right]

        if self.how in ('right', 'outer'):
            left = [[name, option(dt)] for name, dt in left]
        if self.how in ('left', 'outer'):
            right = [[name, option(dt)] for name, dt in right]

        return dshape(Record(joined + left + right))

    @property
    def dshape(self):
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
            fields = fields[0]
        assert fields == expr._name
        return expr.dshape.measure


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
        raise ValueError("Can not Join.  No shared columns between %s and %s" %
                         (lhs, rhs))
    if types_of_fields(on_left, lhs) != types_of_fields(on_right, rhs):
        raise TypeError("Schema's of joining columns do not match")
    _on_left = tuple(on_left) if isinstance(on_left, list) else on_left
    _on_right = (tuple(on_right) if isinstance(on_right, list)
                 else on_right)

    how = how.lower()
    if how not in ('inner', 'outer', 'left', 'right'):
        raise ValueError("How parameter should be one of "
                         "\n\tinner, outer, left, right."
                         "\nGot: %s" % how)

    return Join(lhs, rhs, _on_left, _on_right, how, suffixes)


join.__doc__ = Join.__doc__


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

    @property
    def dshape(self):
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


concat.__doc__ = Concat.__doc__


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

    @property
    def schema(self):
        return datashape.bool_

    def __str__(self):
        return '%s.%s(%s)' % (self._child, type(self).__name__.lower(),
                              self._keys)


def isin(expr, keys):
    if isinstance(keys, Expr):
        raise TypeError('keys argument cannot be an expression, '
                        'it must be an iterable object such as a list, '
                        'tuple or set')
    return IsIn(expr, frozenset(keys))


isin.__doc__ = IsIn.__doc__


dshape_method_list.extend([
    (iscollection, set([sort, head])),
    (lambda ds: len(ds.shape) == 1, set([distinct])),
    (lambda ds: len(ds.shape) == 1 and isscalar(ds.measure), set([isin])),
])
