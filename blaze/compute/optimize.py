from ..dispatch import dispatch
from ..expr import Selection, By

def optimize_traverse(func, expr, new=False, scope=None):
    if new:
        leaves = expr._leaves()
        expr = func(expr, *[v for e, v in scope.items() if e in expr])
    if isinstance(expr, Selection):
        predicate = optimize_traverse(func, expr.predicate, new=True,
                                      scope=scope)
        expr = expr._subs({expr.predicate: predicate})

    if isinstance(expr, By):
        grouper = optimize_traverse(func, expr.grouper, new=True, scope=scope)
        apply = optimize_traverse(func, expr.apply, new=True, scope=scope)
        expr = expr._subs({expr.grouper: grouper,
                           expr.apply: apply})

    if len(expr._inputs) > 1:
        children = [optimize_traverse(func, i, new=True, scope=scope)
                        for i in expr._inputs]
    else:
        children = [optimize_traverse(func, i, scope=scope) for i in expr._inputs]

    return expr._subs(dict(zip(expr._inputs, children)))
