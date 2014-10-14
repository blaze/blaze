from __future__ import absolute_import, division, print_function

from toolz import isdistinct, frequencies, concat, unique, get
import datashape
from datashape import Option, Record, Unit, dshape, var
from datashape.predicates import isscalar, iscollection

from .core import common_subexpression
from .expressions import Expr, ElemWise, Symbol

__all__ = ['Sort', 'Distinct', 'Head', 'Merge', 'Union', 'distinct', 'merge',
           'union', 'head', 'sort', 'Join', 'join']

class Sort(Expr):
    """ Table in sorted order

    Examples
    --------

    >>> accounts = Symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.sort('amount', ascending=False).schema
    dshape("{ name : string, amount : int32 }")

    Some backends support sorting by arbitrary rowwise tables, e.g.

    >>> accounts.sort(-accounts['amount']) # doctest: +SKIP
    """
    __slots__ = '_child', '_key', 'ascending'

    @property
    def dshape(self):
        return self._child.dshape

    @property
    def key(self):
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
            A Table Expression, ``t.sort(-t['amount'])``
    ascending: bool
        Determines order of the sort
    """
    if isinstance(key, list):
        key = tuple(key)
    if key is None:
        key = child.fields[0]
    return Sort(child, key, ascending)


class Distinct(Expr):
    """
    Removes duplicate rows from the table, so every row is distinct

    Examples
    --------

    >>> t = Symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> e = distinct(t)

    >>> data = [('Alice', 100, 1),
    ...         ('Bob', 200, 2),
    ...         ('Alice', 100, 1)]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 100, 1), ('Bob', 200, 2)]
    """
    __slots__ = '_child',

    @property
    def dshape(self):
        return self._child.dshape

    @property
    def fields(self):
        return self._child.fields

    @property
    def _name(self):
        return self._child._name


def distinct(expr):
    return Distinct(expr)


class Head(Expr):
    """ First ``n`` elements of collection

    Examples
    --------

    >>> accounts = Symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.head(5).dshape
    dshape("5 * { name : string, amount : int32 }")
    """
    __slots__ = '_child', 'n'

    @property
    def dshape(self):
        return self.n * self._child.dshape.subshape[0]

    def _len(self):
        return min(self._child._len(), self.n)

    @property
    def _name(self):
        return self._child._name


def head(child, n=10):
    return Head(child, n)

head.__doc__ = Head.__doc__


def merge(*exprs):
    # Get common sub expression
    try:
        child = common_subexpression(*exprs)
    except:
        raise ValueError("No common sub expression found for input expressions")

    result = Merge(child, exprs)

    if not isdistinct(result.fields):
        raise ValueError("Repeated columns found: " + ', '.join(k for k, v in
            frequencies(result.fields).items() if v > 1))

    return result


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
    """ Merge the columns of many Tables together

    Must all descend from same table via ElemWise operations

    Examples
    --------

    >>> accounts = Symbol('accounts', 'var * {name: string, amount: int}')

    >>> newamount = (accounts['amount'] * 1.5).label('new_amount')

    >>> merge(accounts, newamount).fields
    ['name', 'amount', 'new_amount']

    See Also
    --------

    blaze.expr.collections.Union
    blaze.expr.collections.Join
    """
    __slots__ = '_child', 'children'

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


class Union(Expr):
    """ Merge the rows of many Tables together

    Must all have the same schema

    Examples
    --------

    >>> usa_accounts = Symbol('accounts', 'var * {name: string, amount: int}')
    >>> euro_accounts = Symbol('accounts', 'var * {name: string, amount: int}')

    >>> all_accounts = union(usa_accounts, euro_accounts)
    >>> all_accounts.fields
    ['name', 'amount']

    See Also
    --------

    blaze.expr.collections.Merge
    """
    __slots__ = 'children',
    __inputs__ = 'children',

    def _subterms(self):
        yield self
        for i in self.children:
            for node in i._subterms():
                yield node

    @property
    def dshape(self):
        return datashape.var * self.children[0].dshape.subshape[0]

    def _leaves(self):
        return list(unique(concat(i._leaves() for i in self.children)))


def union(*children):
    schemas = set(child.schema for child in children)
    if len(schemas) != 1:
        raise ValueError("Inconsistent schemas:\n\t%s" %
                            '\n\t'.join(map(str, schemas)))
    return Union(children)


def unpack(l):
    """ Unpack items from collections of length 1

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

    >>> names = Symbol('names', 'var * {name: string, id: int}')
    >>> amounts = Symbol('amounts', 'var * {amount: int, id: int}')

    Join tables based on shared column name
    >>> joined = join(names, amounts, 'id')

    Join based on different column names
    >>> amounts = Symbol('amounts', 'var * {amount: int, acctNumber: int}')
    >>> joined = join(names, amounts, 'id', 'acctNumber')

    See Also
    --------

    blaze.expr.collections.Merge
    blaze.expr.collections.Union
    """
    __slots__ = 'lhs', 'rhs', '_on_left', '_on_right', 'how'
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

        >>> t = Symbol('t', 'var * {name: string, amount: int}')
        >>> s = Symbol('t', 'var * {name: string, id: int}')

        >>> join(t, s).schema
        dshape("{ name : string, amount : int32, id : int32 }")

        >>> join(t, s, how='left').schema
        dshape("{ name : string, amount : int32, id : ?int32 }")
        """
        option = lambda dt: dt if isinstance(dt, Option) else Option(dt)

        joined = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                        if name in self.on_left]

        left = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                           if name not in self.on_left]

        right = [[name, dt] for name, dt in self.rhs.schema[0].parameters[0]
                            if name not in self.on_right]

        if self.how in ('right', 'outer'):
            left = [[name, option(dt)] for name, dt in left]
        if self.how in ('left', 'outer'):
            right = [[name, option(dt)] for name, dt in right]

        return dshape(Record(joined + left + right))


    @property
    def dshape(self):
        # TODO: think if this can be generalized
        return var * self.schema


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
    if get(on_left, lhs.schema[0]) != get(on_right, rhs.schema[0]):
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
    (iscollection, set([distinct, head, sort, head])),
    ])
