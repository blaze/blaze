from blaze.compute.optimize import *
from blaze.expr import *


def test_traverse_optimize():
    a = Symbol('a', 'var * {x: int, y: int}')
    b = Symbol('b', 'var * {x: int, z: int}')

    expr = join(a.distinct(), b[b.z>0]).head().y

    log = []
    def ident(expr, *data):
        log.append(expr)
        return expr

    expr2 = optimize_traverse(ident, expr, new=True, scope={a: [], b: []})
    expected = [expr, a.distinct(), b[b.z > 0], b.z > 0]

    assert map(hash, log) == map(hash, expected)
