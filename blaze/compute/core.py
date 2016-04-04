from __future__ import absolute_import, division, print_function

from collections import defaultdict, Iterator, Mapping
from datetime import date, datetime, timedelta
import itertools
import numbers
import warnings

import toolz
from toolz import first, unique, assoc
from toolz.utils import no_default
import pandas as pd
from odo import odo

from ..compatibility import basestring
from ..expr import Expr, Field, Symbol, symbol, Join, Cast
from ..dispatch import dispatch
from ..interactive import coerce_core, into


__all__ = ['compute', 'compute_up']

base = numbers.Number, basestring, date, datetime, timedelta, type(None)


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


@dispatch(Cast, object)
def compute_up(c, b, **kwargs):
    # cast only works on the expression system and does not affect the
    # computation
    return b


@dispatch(base)
def compute_up(a, **kwargs):
    return a


@dispatch((list, tuple))
def compute_up(seq, scope=None, **kwargs):
    return type(seq)(compute(item, scope or {}, **kwargs) for item in seq)


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

    Examples
    --------

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
        scope3 = dict((e, pre_compute_(e, datum,
                                       **assoc(kwargs, 'scope', scope2)))
                      for e, datum in scope2.items())
    else:
        scope3 = scope2
    if optimize_:
        try:
            expr3 = optimize_(expr2, *[scope3[leaf]
                                       for leaf in expr2._leaves()])
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
                                  "type(expr): %s\n"
                                  "expr: %s\n"
                                  "data: %s" % (type(expr3), expr3, scope4))
    else:
        return top_then_bottom_then_top_again_etc(expr3, scope4, **kwargs)


_names = ('leaf_%d' % i for i in itertools.count(1))

_leaf_cache = {}
_used_tokens = defaultdict(set)


def _reset_leaves():
    _leaf_cache.clear()
    _used_tokens.clear()


def makeleaf(expr):
    """ Name of a new leaf replacement for this expression

    >>> _reset_leaves()

    >>> t = symbol('t', '{x: int, y: int, z: int}')
    >>> makeleaf(t) == t
    True
    >>> makeleaf(t.x)
    <`x` symbol; dshape='int32'>
    >>> makeleaf(t.x + 1)
    <`x` symbol; dshape='int64'>
    >>> makeleaf(t.y + 1)
    <`y` symbol; dshape='int64'>
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
    if expr in _leaf_cache:
        return _leaf_cache[expr]
    if isinstance(expr, Symbol):  # Idempotent on symbols
        _used_tokens[name].add(expr._token)
        _leaf_cache[expr] = expr
        return expr
    used_for_name = _used_tokens[name]
    for token in itertools.count():
        if token not in used_for_name:
            break
    result = symbol(name, expr.dshape, token)
    used_for_name.add(token)
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
    (amount_sum + 1, {<`amount_sum` symbol; dshape='int64'>: 600})
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
    new_expr = expr._subs({
        i: e for i, e in zip(inputs, exprs) if not i.isidentical(e)
    })

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


def swap_resources_into_scope(expr, scope):
    """ Translate interactive expressions into normal abstract expressions

    Interactive Blaze expressions link to data on their leaves.  From the
    expr/compute perspective, this is a hack.  We push the resources onto the
    scope and return simple unadorned expressions instead.

    Examples
    --------

    >>> from blaze import data
    >>> t = data([1, 2, 3], dshape='3 * int', name='t')
    >>> swap_resources_into_scope(t.head(2), {})
    (t.head(2), {<`t` symbol; dshape='3 * int32'>: [1, 2, 3]})

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


@dispatch(Expr, Mapping)
def compute(expr, d, return_type=no_default, **kwargs):
    """Compute expression against data sources.

    Parameters
    ----------
    expr : str
        blaze expression
    d: resource
        data source to compute expression on
    return_type : {'native', 'core', type}, optional
        Type to return data as. Defaults to 'native' but will be changed
        to 'core' in version 0.11.  'core' forces the computation into a core type.
        'native' returns the result as is from the respective backend's `post_compute`.
        If a type is passed, it will odo the result into the type before returning.

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
        d3 = dict(
            (e, pre_compute_(e, dat, **kwargs))
            for e, dat in d2.items()
            if e in expr2
        )
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

    # return the backend's native response
    if return_type is no_default:
        msg = ("The default behavior of compute will change in version >= 0.11"
               " where the `return_type` parameter will default to 'core'.")
        warnings.warn(msg, DeprecationWarning)
    # return result as a core type
    # (python type, pandas Series/DataFrame, numpy array)
    elif return_type == 'core':
        result = coerce_core(result, expr.dshape)
    # user specified type
    elif isinstance(return_type, type):
        result = into(return_type, result)
    elif return_type != 'native':
        raise ValueError(
            "Invalid return_type passed to compute: {}".format(return_type),
        )

    return result


@compute.register(Expr, object)
def compute_single_object(expr, o, **kwargs):
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


@dispatch(Field, Mapping)
def compute_up(expr, data, **kwargs):
    return data[expr._name]


@compute_up.register(Join, object, object)
def join_dataframe_to_selectable(expr, lhs, rhs, scope=None, **kwargs):
    lexpr, rexpr = expr._leaves()
    return compute(
        expr,
        {
            lexpr: odo(lhs, pd.DataFrame, dshape=lexpr.dshape),
            rexpr: odo(rhs, pd.DataFrame, dshape=rexpr.dshape)
        },
        **kwargs
    )
