from datashape import Option, dshape, DataShape, Var
from datashape.predicates import iscollection
from .expressions import ElemWise, Expr


__all__ = ['IsNull', 'isnull', 'DropNA', 'dropna']


class IsNull(ElemWise):
    __slots__ = '_hash', '_child',

    @property
    def schema(self):
        return dshape('bool')

    @property
    def dshape(self):
        return DataShape(Var(), self.schema)

    def __str__(self):
        return '%s.isnull()' % self._child


def isnull(expr):
    """Return an indicator as to whether one or more elements is null.

    Parameters
    ----------
    expr : Expr
        A ``blaze`` expression

    Examples
    --------
    >>> from blaze import symbol
    >>> s = symbol('s', 'var * {name: ?string, amount: float64}')
    >>> isnull = s.name.isnull()
    >>> isnull
    s.name.isnull()
    >>> isnull.dshape
    dshape("var * bool")

    Notes
    -----
    * Requires that ``expr.dshape.measure`` be a ``datashape.Option`` type

    See Also
    --------
    dropna
    """
    return IsNull(expr)


class DropNA(Expr):
    __slots__ = '_hash', '_child', 'how'

    @property
    def dshape(self):
        ds = self._child.dshape
        return DataShape(Var(), ds.measure.ty)

    def __str__(self):
        return '%s.dropna(how=%r)' % (self._child, self.how)


def dropna(expr, how='any'):
    """Return a collection with null elements removed.

    Parameters
    ----------
    expr : Expr
        A ``blaze`` expression
    how : str, {``'any'``, ``'all'``}
        If ``'any'``, then drop if any values are null, if ``'all'`` then drop
        only if all values are null. Useful when calling ``dropna`` on an entire
        table.

    Examples
    --------
    >>> from blaze import symbol
    >>> s = symbol('s', 'var * {name: ?string}')
    >>> nonnull = s.name.dropna()
    >>> nonnull
    s.name.dropna(how='any')
    >>> nonnull.dshape
    dshape("var * string")

    Notes
    -----
    * Requires that ``expr.dshape.measure`` be a ``datashape.Option`` type
    * The resulting dshape of the expression is no longer an ``Option`` type

    See Also
    --------
    isnull
    """
    return DropNA(expr, how=how)


from .expressions import schema_method_list, dshape_method_list


schema_method_list.extend([
    (lambda ds: isinstance(ds, Option), set([isnull]))
])


dshape_method_list.extend([
    (lambda ds: isinstance(ds.measure, Option) and iscollection(ds),
     set([dropna])),
])
