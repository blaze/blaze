from blaze.expr.optimize import lean_projection
from blaze.expr import *

def test_lean_projection():
    t = Symbol('t', 'var * {x: int, y: int, z: int, w: int}')

    assert lean_projection(t[t.x > 0].y)._child._child.isidentical(t[['x', 'y']])


def test_lean_projection_by():
    t = Symbol('t', 'var * {x: int, y: int, z: int, w: int}')

    assert lean_projection(by(t.x, t.y.sum()))._child.isidentical(
                    t[['x', 'y']])
    assert lean_projection(by(t.x, total=t.y.sum()))._child.isidentical(
                    t[['x', 'y']])
