from __future__ import absolute_import, division, print_function

import unittest
import blaze
from blaze.datadescriptor import dd_as_py
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

class TestBitwiseOps(unittest.TestCase):
    def test_bitwise_or_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t | t)._data), True)
        self.assertEqual(dd_as_py((t | f)._data), True)
        self.assertEqual(dd_as_py((f | t)._data), True)
        self.assertEqual(dd_as_py((f | f)._data), False)

    def test_bitwise_or_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a | b)._data), x | y)
        self.assertEqual(dd_as_py(blaze.bitwise_or(a, b)._data), x | y)

    def test_bitwise_and_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t & t)._data), True)
        self.assertEqual(dd_as_py((t & f)._data), False)
        self.assertEqual(dd_as_py((f & t)._data), False)
        self.assertEqual(dd_as_py((f & f)._data), False)

    def test_bitwise_and_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a & b)._data), x & y)
        self.assertEqual(dd_as_py(blaze.bitwise_and(a, b)._data), x & y)

    def test_bitwise_xor_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t ^ t)._data), False)
        self.assertEqual(dd_as_py((t ^ f)._data), True)
        self.assertEqual(dd_as_py((f ^ t)._data), True)
        self.assertEqual(dd_as_py((f ^ f)._data), False)

    def test_bitwise_xor_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a ^ b)._data), x ^ y)
        self.assertEqual(dd_as_py(blaze.bitwise_xor(a, b)._data), x ^ y)

    def test_bitwise_not_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((~t)._data), False)
        self.assertEqual(dd_as_py((~f)._data), True)

    def test_bitwise_not_uint64(self):
        x = 0x3192573469a2b3a1
        a = blaze.array(x, 'uint64')
        self.assertEqual(dd_as_py((~a)._data), x ^ 0xffffffffffffffff)
        self.assertEqual(dd_as_py(blaze.bitwise_not(a)._data),
                         x ^ 0xffffffffffffffff)

class TestLogAddExp(unittest.TestCase):
    def test_logaddexp_values(self) :
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for ds, dec in zip(['float32', 'float64'], [6, 15]) :
            xf = blaze.log(blaze.array(x, dshape=ds))
            yf = blaze.log(blaze.array(y, dshape=ds))
            zf = blaze.log(blaze.array(z, dshape=ds))
            result = blaze.eval(blaze.logaddexp(xf, yf))
            assert_almost_equal(np.array(result), np.array(zf), decimal=dec)

    def test_logaddexp_range(self) :
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for ds in ['float32', 'float64'] :
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.eval(blaze.logaddexp(logxf, logyf))
            assert_almost_equal(np.array(result), np.array(logzf))

    def test_inf(self) :
        inf = blaze.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        for ds in ['float32', 'float64'] :
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.eval(blaze.logaddexp(logxf, logyf))
            assert_equal(np.array(result), np.array(logzf))

    def test_nan(self):
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, blaze.inf)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.inf, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, 0)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(0, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, blaze.nan)))
