from __future__ import absolute_import, division, print_function
import numbers
from datetime import date, datetime
import toolz
from toolz import first

from ..compatibility import basestring
from ..expr import Expr, Symbol, Symbol, eval_str, Union
from ..dispatch import dispatch
from .optimize import optimize_traverse

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
def compute_up(seq, scope=None, **kwargs):
    return type(seq)(compute(item, scope or {}, **kwargs) for item in seq)


@dispatch(Expr, object)
def compute(expr, o, **kwargs):
    """ Compute against single input

    Assumes that only one Symbol exists in expression

    >>> t = Symbol('t', 'var * {name: string, balance: int}')
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


def top_to_bottom(d, expr, **kwargs):
    """ Processes an expression top-down then bottom-up """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # See if we have a direct computation path
    if (hasattr(expr, '_leaves') and compute_down.resolve(
            (type(expr),) + tuple([type(d.get(leaf)) for leaf in expr._leaves()]))):
        leaves = [d[leaf] for leaf in expr._leaves()]
        try:
            return compute_down(expr, *leaves, **kwargs)
        except NotImplementedError:
            pass

    # Otherwise...
    # Compute children of this expression
    children = ([top_to_bottom(d, child, **kwargs)
                    for child in expr._inputs]
                    if hasattr(expr, '_inputs') else [])

    # Compute this expression given the children
    return compute_up(expr, *children, scope=d, **kwargs)


def bottom_up(d, expr):
    """
    Process an expression from the leaves upwards

    Parameters
    ----------

    d : dict mapping {Symbol: data}
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


@dispatch(Expr, object)
def pre_compute(leaf, data):
    """ Transform data prior to calling ``compute`` """
    return data


@dispatch(Expr, object, dict)
def post_compute(expr, result, d):
    """ Effects after the computation is complete """
    return result


@dispatch(Expr, object)
def optimize(expr, data):
    """ Optimize expression to be computed on data """
    return expr


def swap_resources_into_scope(expr, scope):
    """ Translate interactive expressions into normal abstract expressions

    Interactive Blaze expressions link to data on their leaves.  From the
    expr/compute perspective, this is a hack.  We push the resources onto the
    scope and return simple unadorned expressions instead.

    Example
    -------

    >>> from blaze import Data
    >>> t = Data([1, 2, 3], dshape='3 * int', name='t')
    >>> swap_resources_into_scope(t.head(2), {})
    (t.head(2), {t: [1, 2, 3]})
    """
    resources = expr._resources()
    symbol_dict = dict((t, Symbol(t._name, t.dshape)) for t in resources)
    resources = dict((symbol_dict[k], v) for k, v in resources.items())
    scope = toolz.merge(resources, scope)
    expr = expr._subs(symbol_dict)

    return expr, scope


@dispatch(Expr, dict)
def compute(expr, d, **kwargs):
    """ Compute expression against data sources

    >>> t = Symbol('t', 'var * {name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> list(compute(deadbeats, {t: data}))
    ['Bob', 'Charlie']
    """
    expr2, d2 = swap_resources_into_scope(expr, d)
    d3 = dict((e, pre_compute(e, dat)) for e, dat in d2.items())

    try:
        expr3 = optimize_traverse(optimize, expr2, new=True, scope=d3)
    except NotImplementedError:
        expr3 = expr2
    result = top_to_bottom(d3, expr3, **kwargs)
    return post_compute(expr3, result, d3)


@dispatch(Union, (list, tuple))
def compute_up(t, children, **kwargs):
    return compute_up(t, children[0], tuple(children))
