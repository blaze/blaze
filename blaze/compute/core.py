from __future__ import absolute_import, division, print_function
import numbers
from datetime import date, datetime
import toolz
from toolz import first

from ..compatibility import basestring
from ..expr import Expr, Symbol, TableSymbol, eval_str, Union
from ..dispatch import dispatch

__all__ = ['compute', 'compute_up']

base = (numbers.Real, basestring, date, datetime)


@dispatch(object, object)
def compute_up(a, b, **kwargs):
    raise NotImplementedError("Blaze does not know how to compute "
                              "expression of type `%s` on data of type `%s`"
                              % (type(a).__name__, type(b).__name__))


@dispatch(base)
def compute_up(a, **kwargs):
    return a


@dispatch((list, tuple))
def compute_up(seq, scope={}, **kwargs):
    return type(seq)(compute(item, scope, **kwargs) for item in seq)


@dispatch(Expr, object)
def compute(expr, o, **kwargs):
    """ Compute against single input

    Assumes that only one TableSymbol exists in expression

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> # list(compute(deadbeats, {t: data}))
    >>> list(compute(deadbeats, data))
    ['Bob', 'Charlie']
    """
    ts = set([x for x in expr._subterms() if isinstance(x, Symbol)])
    if len(ts) == 1:
        return compute(expr, {first(ts): o}, **kwargs)
    else:
        raise ValueError("Give compute dictionary input, got %s" % str(o))


@dispatch(object)
def compute_down(expr):
    """ Compute the expression on the entire inputs

    inputs match up to leaves of the expression
    """
    return expr


def top_to_bottom(d, expr):
    """ Processes an expression top-down then bottom-up """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # See if we have a direct computation path
    if (hasattr(expr, '_leaves') and compute_down.resolve(
            (type(expr),) + tuple(type(d.get(leaf)) for leaf in expr._leaves()))):
        leaves = [d[leaf] for leaf in expr._leaves()]
        return compute_down(expr, *leaves)
    else:
        # Compute children of this expression
        children = ([top_to_bottom(d, child) for child in expr._inputs]
                    if hasattr(expr, '_inputs') else [])

        # Compute this expression given the children
        return compute_up(expr, *children, scope=d)


def bottom_up(d, expr):
    """
    Process an expression from the leaves upwards

    Parameters
    ----------

    d : dict mapping {TableSymbol: data}
        Maps expressions to data elements, likely at the leaves of the tree
    expr : Expr
        Expression to compute

    Helper function for ``compute``
    """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # Compute children of this expression
    children = ([bottom_up(d, child) for child in expr._inputs]
                if hasattr(expr, '_inputs') else [])

    # Compute this expression given the children
    result = compute_up(expr, *children, scope=d)

    return result


@dispatch(Expr, dict)
def pre_compute(expr, d):
    """ Transform expr prior to calling ``compute`` """
    return expr


@dispatch(Expr, object, dict)
def post_compute(expr, result, d):
    """ Effects after the computation is complete """
    return result


def swap_resources_into_scope(expr, scope):
    """ Translate interactive expressions into normal abstract expressions

    Interactive Blaze expressions link to data on their leaves.  From the
    expr/compute perspective, this is a hack.  We push the resources onto the
    scope and return simple unadorned expressions instead.

    Example
    -------

    >>> from blaze import Table
    >>> t = Table([1, 2, 3], dshape='3 * int', name='t')
    >>> swap_resources_into_scope(t, {})
    (t, {t: [1, 2, 3]})
    """
    resources = expr.resources()
    symbol_dict = dict((t, Symbol(t._name, t.dshape)) for t in resources)
    resources = {symbol_dict[k]: v for k, v in resources.items()}
    scope = toolz.merge(resources, scope)
    expr = expr._subs(symbol_dict)

    return expr, scope


@dispatch(Expr, dict)
def compute(expr, d):
    """ Compute expression against data sources

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> list(compute(deadbeats, {t: data}))
    ['Bob', 'Charlie']
    """
    expr, d = swap_resources_into_scope(expr, d)

    expr = pre_compute(expr, d)
    result = top_to_bottom(d, expr)
    return post_compute(expr, result, d)


def columnwise_funcstr(t, variadic=True, full=False):
    """Build a string that can be eval'd to return a ``lambda`` expression.

    Parameters
    ----------
    t : Broadcast
        An expression whose leaves (at each application of the returned
        expression) are all instances of ``ScalarExpression``.
        For example ::

            t.petal_length / max(t.petal_length)

        is **not** a valid ``Broadcast``, since the expression ::

            max(t.petal_length)

        has a leaf ``t`` that is not a ``ScalarExpression``. A example of a
        valid ``Broadcast`` expression is ::

            t.petal_length / 4

    Returns
    -------
    f : str
        A string that can be passed to ``eval`` and will return a function that
        operates on each row and applies a scalar expression to a subset of the
        columns in each row.

    Examples
    --------
    >>> t = TableSymbol('t', '{x: real, y: real, z: real}')
    >>> cw = t['x'] + t['z']
    >>> columnwise_funcstr(cw)
    'lambda x, z: x + z'

    >>> columnwise_funcstr(cw, variadic=False)
    'lambda (x, z): x + z'

    >>> columnwise_funcstr(cw, variadic=False, full=True)
    'lambda (x, y, z): x + z'
    """
    if full:
        columns = t._child.fields
    else:
        columns = t.active_columns()
    if variadic:
        prefix = 'lambda %s: '
    else:
        prefix = 'lambda (%s): '

    return prefix % ', '.join(map(str, columns)) + eval_str(t.expr)


@dispatch(Union, (list, tuple))
def compute_up(t, children, **kwargs):
    return compute_up(t, children[0], tuple(children))
