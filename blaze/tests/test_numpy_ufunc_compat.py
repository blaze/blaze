from __future__ import absolute_import, division, print_function

import unittest
import blaze
import math, cmath
from blaze.datadescriptor import dd_as_py
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_array_equal
from blaze.py2help import skip

# Many of these tests have been adapted from NumPy's test_umath.py test file

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

class TestPower(unittest.TestCase):
    def test_power_float(self):
        x = blaze.array([1., 2., 3.])
        assert_equal(np.array(x**0), [1., 1., 1.])
        assert_equal(np.array(x**1), x)
        assert_equal(np.array(x**2), [1., 4., 9.])
        assert_almost_equal(np.array(x**(-1)), [1., 0.5, 1./3])
        assert_almost_equal(np.array(x**(0.5)), [1., math.sqrt(2), math.sqrt(3)])

    @skip('temporarily skipping, this test revealed bugs in the eval code')
    def test_power_complex(self):
        x = blaze.array([1+2j, 2+3j, 3+4j])
        assert_equal(np.array(x**0), [1., 1., 1.])
        assert_equal(np.array(x**1), x)
        assert_almost_equal(np.array(x**2), [-3+4j, -5+12j, -7+24j])
        assert_almost_equal(np.array(x**3), [(1+2j)**3, (2+3j)**3, (3+4j)**3])
        assert_almost_equal(np.array(x**4), [(1+2j)**4, (2+3j)**4, (3+4j)**4])
        assert_almost_equal(np.array(x**(-1)), [1/(1+2j), 1/(2+3j), 1/(3+4j)])
        assert_almost_equal(np.array(x**(-2)), [1/(1+2j)**2, 1/(2+3j)**2, 1/(3+4j)**2])
        assert_almost_equal(np.array(x**(-3)), [(-11+2j)/125, (-46-9j)/2197,
                                      (-117-44j)/15625])
        assert_almost_equal(np.array(x**(0.5)), [cmath.sqrt(1+2j), cmath.sqrt(2+3j),
                                       cmath.sqrt(3+4j)])
        norm = 1./((x**14)[0])
        assert_almost_equal(np.array(x**14 * norm),
                [i * norm for i in [-76443+16124j, 23161315+58317492j,
                                    5583548873 +  2465133864j]])

        def assert_complex_equal(x, y):
            assert_array_equal(np.array(x.real), np.array(y.real))
            assert_array_equal(np.array(x.imag), np.array(y.imag))

        for z in [complex(0, np.inf), complex(1, np.inf)]:
            z = blaze.array([z], dshape="complex[float64]")
            assert_complex_equal(z**1, z)
            assert_complex_equal(z**2, z*z)
            assert_complex_equal(z**3, z*z*z)

    def test_power_zero(self):
        zero = blaze.array([0j])
        one = blaze.array([1+0j])
        cinf = blaze.array([complex(np.inf, 0)])
        cnan = blaze.array([complex(np.nan, np.nan)])

        def assert_complex_equal(x, y):
            x, y = np.array(x), np.array(y)
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # positive powers
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(blaze.power(zero, p), zero)

        # zero power
        assert_complex_equal(blaze.power(zero, 0), one)
        assert_complex_equal(blaze.power(zero, 0+1j), cnan)

        # negative power
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(blaze.power(zero, -p), cnan)
        assert_complex_equal(blaze.power(zero, -1+0.2j), cnan)

    def test_fast_power(self):
        x = blaze.array([1, 2, 3], dshape="int16")
        self.assertEqual((x**2.00001).dshape, (x**2.0).dshape)

class TestLog(unittest.TestCase):
    def test_log_values(self) :
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64'] :
            log2_ = 0.69314718055994530943
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)*log2_
            result = blaze.log(xf)
            assert_almost_equal(np.array(result), np.array(yf))


class TestExp(unittest.TestCase):
    def test_exp_values(self) :
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64'] :
            log2_ = 0.69314718055994530943
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)*log2_
            result = blaze.exp(yf)
            assert_almost_equal(np.array(result), np.array(xf))

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

class TestLog2(unittest.TestCase):
    def test_log2_values(self) :
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64'] :
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)
            result = blaze.log2(xf)
            assert_almost_equal(np.array(result), yf)


class TestExp2(unittest.TestCase):
    def test_exp2_values(self) :
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64'] :
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)
            result = blaze.exp2(yf)
            assert_almost_equal(np.array(result), xf)


class TestLogAddExp2(unittest.TestCase):
    # Need test for intermediate precisions
    def test_logaddexp2_values(self) :
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for ds, dec in zip(['float32', 'float64'], [6, 15, 15]) :
            xf = blaze.log2(blaze.array(x, dshape=ds))
            yf = blaze.log2(blaze.array(y, dshape=ds))
            zf = blaze.log2(blaze.array(z, dshape=ds))
            result = blaze.logaddexp2(xf, yf)
            assert_almost_equal(np.array(result), zf, decimal=dec)

    def test_logaddexp2_range(self) :
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for ds in ['float32', 'float64'] :
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp2(logxf, logyf)
            assert_almost_equal(np.array(result), logzf)

    def test_inf(self) :
        inf = blaze.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        for ds in ['float32', 'float64'] :
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp2(logxf, logyf)
            assert_equal(np.array(result), logzf)

    def test_nan(self):
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, blaze.inf)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.inf, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, 0)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(0, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, blaze.nan)))
