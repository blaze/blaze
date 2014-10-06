
from .core import common_subexpression
from .expressions import Expr, ElemWise, Symbol
from datashape import Option, Record, Unit, dshape
from datashape.predicates import isscalar, iscollection
import datashape
from toolz import isdistinct, frequencies, concat, unique

__all__ = ['Sort', 'Distinct', 'Head', 'Merge', 'Union', 'distinct', 'merge',
'union', 'head', 'sort']

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

    blaze.expr.table.Union
    blaze.expr.table.Join
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

    blaze.expr.table.Merge
    blaze.expr.table.Join
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

from .expressions import dshape_method_list

dshape_method_list.extend([
    (iscollection, set([distinct, head, sort, head])),
    ])
