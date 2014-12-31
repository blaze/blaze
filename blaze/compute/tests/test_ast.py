import numpy as np
from blaze import discover
from blaze.compute.ast import lambdify
from blaze import cos, symbol


def test_simple():
    x = np.array([1.0, 2.0, 3.0])
    t = symbol('t', discover(x))
    f = lambdify([t], cos(t))
    result = f(x)
    expected = np.cos(x)
    np.testing.assert_array_equal(result, expected)
