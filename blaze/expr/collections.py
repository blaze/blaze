from __future__ import absolute_import, division, print_function

from toolz import isdistinct, frequencies, concat, unique, get
import datashape
from datashape import Option, Record, Unit, dshape, var
from datashape.predicates import isscalar, iscollection, isrecord

from .core import common_subexpression
from .expressions import Expr, ElemWise, label

__all__ = ['Sort', 'Distinct', 'Head', 'Merge', 'distinct', 'merge',
           'head', 'sort', 'Join', 'join', 'transform']

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
    """ Sort collection

    Parameters
    ----------
    key: string, list of strings, Expr
        Defines by what you want to sort.  Either:
            A single column string, ``t.sort('amount')``
            A list of column strings, ``t.sort(['name', 'amount'])``
            A Table Expression, ``t.sort(-t.amount)``
    ascending: bool
        Determines order of the sort
    """
    if not isrecord(child.dshape.measure):
        key = None
    if isinstance(key, list):
        key = tuple(key)
    return Sort(child, key, ascending)


class Distinct(Expr):
    """
    Removes duplicate rows from the table, so every row is distinct

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


class Head(Expr):
    """ First ``n`` elements of collection

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
    exprs = exprs + tuple(label(v, k) for k, v in kwargs.items())
    try:
        child = common_subexpression(*exprs)
    except:
        raise ValueError("No common sub expression found for input expressions")

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

    args = [t] + [v.label(k) for k, v in kwargs.items()]
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
        return list(concat(child.fields for child in self.children))

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
        return list(unique(concat(i._leaves() for i in self.children)))


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
    lhs : Expr
    rhs : Expr
    on_left : string
    on_right : string

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
    __slots__ = '_hash', 'lhs', 'rhs', '_on_left', '_on_right', 'how'
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
                zip(self.lhs.fields, types_of_fields(self.lhs.fields, self.lhs))
                           if name not in self.on_left]

        right = [[name, dt] for name, dt in
                zip(self.rhs.fields, types_of_fields(self.rhs.fields, self.rhs))
                           if name not in self.on_right]


        # Handle overlapping but non-joined case, e.g.
        left_other  = [name for name, dt in left  if name not in self.on_left]
        right_other = [name for name, dt in right if name not in self.on_right]
        overlap = set.intersection(set(left_other), set(right_other))
        left = [[name+'_left' if name in overlap else name, dt]
                for name, dt in left]
        right = [[name+'_right' if name in overlap else name, dt]
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


def join(lhs, rhs, on_left=None, on_right=None, how='inner'):
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
        raise ValueError("Can not Join.  No shared columns between %s and %s"%
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

    return Join(lhs, rhs, _on_left, _on_right, how)


join.__doc__ = Join.__doc__


from .expressions import dshape_method_list

dshape_method_list.extend([
    (iscollection, set([sort, head])),
    (lambda ds: len(ds.shape) == 1, set([distinct])),
    ])
