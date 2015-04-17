from __future__ import absolute_import, division, print_function

import numbers
from datetime import date, datetime
import toolz
from toolz import first, concat, memoize, unique, assoc
import itertools
from collections import Iterator

from ..compatibility import basestring
from ..expr import Expr, Field, Symbol, symbol, eval_str
from ..dispatch import dispatch

__all__ = ['compute', 'compute_up']

base = (numbers.Number, basestring, date, datetime)


@dispatch(Expr, object)
def pre_compute(leaf, data, scope=None, **kwargs):
    """ Transform data prior to calling ``compute`` """
    return data


@dispatch(Expr, object)
def post_compute(expr, result, scope=None):
    """ Effects after the computation is complete """
    return result


@dispatch(Expr, object)
def optimize(expr, data):
    """ Optimize expression to be computed on data """
    return expr


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

    >>> t = symbol('t', 'var * {name: string, balance: int}')
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
def compute_down(expr, **kwargs):
    """ Compute the expression on the entire inputs

    inputs match up to leaves of the expression
    """
    return expr


def issubtype(a, b):
    """ A custom issubclass """
    if issubclass(a, b):
        return True
    if issubclass(a, (tuple, list, set)) and issubclass(b, Iterator):
        return True
    if issubclass(b, (tuple, list, set)) and issubclass(a, Iterator):
        return True
    return False

def type_change(old, new):
    """ Was there a significant type change between old and new data?

    >>> type_change([1, 2], [3, 4])
    False
    >>> type_change([1, 2], [3, [1,2,3]])
    True

    Some special cases exist, like no type change from list to Iterator

    >>> type_change([[1, 2]], [iter([1, 2])])
    False
    """
    if all(isinstance(x, base) for x in old + new):
        return False
    if len(old) != len(new):
        return True
    new_types = list(map(type, new))
    old_types = list(map(type, old))
    return not all(map(issubtype, new_types, old_types))


def top_then_bottom_then_top_again_etc(expr, scope, **kwargs):
    """ Compute expression against scope

    Does the following interpreter strategy:

    1.  Try compute_down on the entire expression
    2.  Otherwise compute_up from the leaves until we experience a type change
        (e.g. data changes from dict -> pandas DataFrame)
    3.  Re-optimize expression and re-pre-compute data
    4.  Go to step 1

    Example
    -------

    >>> import numpy as np

    >>> s = symbol('s', 'var * {name: string, amount: int}')
    >>> data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
    ...                 dtype=[('name', 'S7'), ('amount', 'i4')])

    >>> e = s.amount.sum() + 1
    >>> top_then_bottom_then_top_again_etc(e, {s: data})
    601

    See Also
    --------

    bottom_up_until_type_break  -- uses this for bottom-up traversal
    top_to_bottom -- older version
    bottom_up -- older version still
    """
    # 0. Base case: expression is in dict, return associated data
    if expr in scope:
        return scope[expr]

    if not hasattr(expr, '_leaves'):
        return expr

    leaf_exprs = list(expr._leaves())
    leaf_data = [scope.get(leaf) for leaf in leaf_exprs]

    # 1. See if we have a direct computation path with compute_down
    try:
        return compute_down(expr, *leaf_data, **kwargs)
    except NotImplementedError:
        pass

    # 2. Compute from the bottom until there is a data type change
    expr2, scope2 = bottom_up_until_type_break(expr, scope, **kwargs)

    # 3. Re-optimize data and expressions
    optimize_ = kwargs.get('optimize', optimize)
    pre_compute_ = kwargs.get('pre_compute', pre_compute)
    if pre_compute_:
        scope3 = dict((e, pre_compute_(expr2, datum,
                                       **assoc(kwargs, 'scope', scope2)))
                        for e, datum in scope2.items())
    else:
        scope3 = scope2
    if optimize_:
        try:
            expr3 = optimize_(expr2, *[scope3[leaf] for leaf in expr2._leaves()])
            _d = dict(zip(expr2._leaves(), expr3._leaves()))
            scope4 = dict((e._subs(_d), d) for e, d in scope3.items())
        except NotImplementedError:
            expr3 = expr2
            scope4 = scope3
    else:
        expr3 = expr2
        scope4 = scope3

    # 4. Repeat
    if expr.isidentical(expr3):
        raise NotImplementedError("Don't know how to compute:\n"
                "expr: %s\n"
                "data: %s" % (expr3, scope4))
    else:
        return top_then_bottom_then_top_again_etc(expr3, scope4, **kwargs)


def top_to_bottom(d, expr, **kwargs):
    """ Processes an expression top-down then bottom-up """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    if not hasattr(expr, '_leaves'):
        return expr

    leaves = list(expr._leaves())
    data = [d.get(leaf) for leaf in leaves]

    # See if we have a direct computation path with compute_down
    try:
        return compute_down(expr, *data, **kwargs)
    except NotImplementedError:
        pass

    optimize_ = kwargs.get('optimize', optimize)
    pre_compute_ = kwargs.get('pre_compute', pre_compute)

    # Otherwise...
    # Compute children of this expression
    if hasattr(expr, '_inputs'):
        children = [top_to_bottom(d, child, **kwargs)
                        for child in expr._inputs]
    else:
        children = []

    # Did we experience a data type change?
    if type_change(data, children):

        # If so call pre_compute again
        if pre_compute_:
            children = [pre_compute_(expr, child, **kwargs) for child in children]

        # If so call optimize again
        if optimize_:
            try:
                expr = optimize_(expr, *children)
            except NotImplementedError:
                pass

    # Compute this expression given the children
    return compute_up(expr, *children, scope=d, **kwargs)


_names = ('leaf_%d' % i for i in itertools.count(1))

_leaf_cache = dict()
_used_tokens = set()
def _reset_leaves():
    _leaf_cache.clear()
    _used_tokens.clear()

def makeleaf(expr):
    """ Name of a new leaf replacement for this expression

    >>> _reset_leaves()

    >>> t = symbol('t', '{x: int, y: int, z: int}')
    >>> makeleaf(t)
    t
    >>> makeleaf(t.x)
    x
    >>> makeleaf(t.x + 1)
    x
    >>> makeleaf(t.x + 1)
    x
    >>> makeleaf(t.x).isidentical(makeleaf(t.x + 1))
    False

    >>> from blaze import sin, cos
    >>> x = symbol('x', 'real')
    >>> makeleaf(cos(x)**2).isidentical(sin(x)**2)
    False

    >>> makeleaf(t) is t  # makeleaf passes on Symbols
    True
    """
    name = expr._name or '_'
    token = None
    if expr in _leaf_cache:
        return _leaf_cache[expr]
    if isinstance(expr, Symbol):  # Idempotent on symbols
        return expr
    if (name, token) in _used_tokens:
        for token in itertools.count():
            if (name, token) not in _used_tokens:
                break
    result = symbol(name, expr.dshape, token)
    _used_tokens.add((name, token))
    _leaf_cache[expr] = result
    return result


def data_leaves(expr, scope):
    return [scope[leaf] for leaf in expr._leaves()]


def bottom_up_until_type_break(expr, scope, **kwargs):
    """ Traverse bottom up until data changes significantly

    Parameters
    ----------

    expr: Expression
        Expression to compute
    scope: dict
        namespace matching leaves of expression to data

    Returns
    -------

    expr: Expression
        New expression with lower subtrees replaced with leaves
    scope: dict
        New scope with entries for those leaves

    Examples
    --------

    >>> import numpy as np

    >>> s = symbol('s', 'var * {name: string, amount: int}')
    >>> data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
    ...                 dtype=[('name', 'S7'), ('amount', 'i8')])

    This computation completes without changing type.  We get back a leaf
    symbol and a computational result

    >>> e = (s.amount + 1).distinct()
    >>> bottom_up_until_type_break(e, {s: data}) # doctest: +SKIP
    (amount, {amount: array([101, 201, 301])})

    This computation has a type change midstream (``list`` to ``int``), so we
    stop and get the unfinished computation.

    >>> e = s.amount.sum() + 1
    >>> bottom_up_until_type_break(e, {s: data})
    (amount_sum + 1, {amount_sum: 600})
    """
    # 0. Base case.  Return if expression is in scope
    if expr in scope:
        leaf = makeleaf(expr)
        return leaf, {leaf: scope[expr]}

    inputs = list(unique(expr._inputs))

    # 1. Recurse down the tree, calling this function on children
    #    (this is the bottom part of bottom up)
    exprs, new_scopes = zip(*[bottom_up_until_type_break(i, scope, **kwargs)
                             for i in inputs])

    # 2. Form new (much shallower) expression and new (more computed) scope
    new_scope = toolz.merge(new_scopes)
    new_expr = expr._subs(dict((i, e) for i, e in zip(inputs, exprs)
                                      if not i.isidentical(e)))

    old_expr_leaves = expr._leaves()
    old_data_leaves = [scope.get(leaf) for leaf in old_expr_leaves]

    # 3. If the leaves have changed substantially then stop
    key = lambda x: str(type(x))
    if type_change(sorted(new_scope.values(), key=key),
                   sorted(old_data_leaves, key=key)):
        return new_expr, new_scope
    # 4. Otherwise try to do some actual work
    try:
        leaf = makeleaf(expr)
        _data = [new_scope[i] for i in new_expr._inputs]
    except KeyError:
        return new_expr, new_scope
    try:
        return leaf, {leaf: compute_up(new_expr, *_data, scope=new_scope,
                                       **kwargs)}
    except NotImplementedError:
        return new_expr, new_scope


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

    >>> expr, scope = _
    >>> list(scope.keys())[0]._resources()
    {}
    """
    resources = expr._resources()
    symbol_dict = dict((t, symbol(t._name, t.dshape)) for t in resources)
    resources = dict((symbol_dict[k], v) for k, v in resources.items())
    other_scope = dict((k, v) for k, v in scope.items()
                       if k not in symbol_dict)
    new_scope = toolz.merge(resources, other_scope)
    expr = expr._subs(symbol_dict)

    return expr, new_scope


@dispatch(Expr, dict)
def compute(expr, d, **kwargs):
    """ Compute expression against data sources

    >>> t = symbol('t', 'var * {name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> list(compute(deadbeats, {t: data}))
    ['Bob', 'Charlie']
    """
    _reset_leaves()
    optimize_ = kwargs.get('optimize', optimize)
    pre_compute_ = kwargs.get('pre_compute', pre_compute)
    post_compute_ = kwargs.get('post_compute', post_compute)

    expr2, d2 = swap_resources_into_scope(expr, d)
    if pre_compute_:
        d3 = dict([(e, pre_compute_(expr2, dat, **kwargs))
                        for e, dat in d2.items()
                        if e in expr2])
    else:
        d3 = d2

    if optimize_:
        try:
            expr3 = optimize_(expr2, *[v for e, v in d3.items() if e in expr2])
            _d = dict(zip(expr2._leaves(), expr3._leaves()))
            d4 = dict((e._subs(_d), d) for e, d in d3.items())
        except NotImplementedError:
            expr3 = expr2
            d4 = d3
    else:
        expr3 = expr2
        d4 = d3

    result = top_then_bottom_then_top_again_etc(expr3, d4, **kwargs)
    if post_compute_:
        result = post_compute_(expr3, result, scope=d4)

    return result


@dispatch(Field, dict)
def compute_up(expr, data, **kwargs):
    return data[expr._name]
