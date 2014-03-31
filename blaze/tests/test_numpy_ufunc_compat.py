from __future__ import absolute_import, division, print_function

import math
import cmath
import unittest

import numpy as np
from numpy import testing
from numpy.testing import assert_

import blaze
import datashape
from blaze.datadescriptor import ddesc_as_py
from blaze.py2help import skip


def assert_almost_equal(actual, desired, **kwargs):
    return testing.assert_almost_equal(np.array(actual),
                                       np.array(desired), **kwargs)


def assert_allclose(actual, desired, **kwargs):
    return testing.assert_allclose(np.array(actual),
                                   np.array(desired), **kwargs)


def assert_equal(actual, desired, **kwargs):
    return testing.assert_equal(np.array(actual), np.array(desired), **kwargs)


def assert_array_equal(actual, desired, **kwargs):
    return testing.assert_array_equal(np.array(actual),
                                      np.array(desired), **kwargs)

# Many of these tests have been adapted from NumPy's test_umath.py test file


class TestBitwiseOps(unittest.TestCase):
    def test_bitwise_or_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(ddesc_as_py((t | t)._data), True)
        self.assertEqual(ddesc_as_py((t | f)._data), True)
        self.assertEqual(ddesc_as_py((f | t)._data), True)
        self.assertEqual(ddesc_as_py((f | f)._data), False)

    def test_bitwise_or_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(ddesc_as_py((a | b)._data), x | y)
        self.assertEqual(ddesc_as_py(blaze.bitwise_or(a, b)._data), x | y)

    def test_bitwise_and_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(ddesc_as_py((t & t)._data), True)
        self.assertEqual(ddesc_as_py((t & f)._data), False)
        self.assertEqual(ddesc_as_py((f & t)._data), False)
        self.assertEqual(ddesc_as_py((f & f)._data), False)

    def test_bitwise_and_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(ddesc_as_py((a & b)._data), x & y)
        self.assertEqual(ddesc_as_py(blaze.bitwise_and(a, b)._data), x & y)

    def test_bitwise_xor_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(ddesc_as_py((t ^ t)._data), False)
        self.assertEqual(ddesc_as_py((t ^ f)._data), True)
        self.assertEqual(ddesc_as_py((f ^ t)._data), True)
        self.assertEqual(ddesc_as_py((f ^ f)._data), False)

    def test_bitwise_xor_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(ddesc_as_py((a ^ b)._data), x ^ y)
        self.assertEqual(ddesc_as_py(blaze.bitwise_xor(a, b)._data), x ^ y)

    def test_bitwise_not_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(ddesc_as_py((~t)._data), False)
        self.assertEqual(ddesc_as_py((~f)._data), True)

    def test_bitwise_not_uint64(self):
        x = 0x3192573469a2b3a1
        a = blaze.array(x, 'uint64')
        self.assertEqual(ddesc_as_py((~a)._data), x ^ 0xffffffffffffffff)
        self.assertEqual(ddesc_as_py(blaze.bitwise_not(a)._data),
                         x ^ 0xffffffffffffffff)


class TestPower(unittest.TestCase):
    def test_power_float(self):
        x = blaze.array([1., 2., 3.])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [1., 4., 9.])
        assert_almost_equal(x**(-1), [1., 0.5, 1./3])
        assert_almost_equal(x**(0.5), [1., math.sqrt(2), math.sqrt(3)])

    def test_power_complex(self):
        x = blaze.array([1+2j, 2+3j, 3+4j])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_almost_equal(x**2, [-3+4j, -5+12j, -7+24j])
        assert_almost_equal(x**3, [(1+2j)**3, (2+3j)**3, (3+4j)**3])
        assert_almost_equal(x**4, [(1+2j)**4, (2+3j)**4, (3+4j)**4])
        assert_almost_equal(x**(-1), [1/(1+2j), 1/(2+3j), 1/(3+4j)])
        assert_almost_equal(x**(-2), [1/(1+2j)**2, 1/(2+3j)**2, 1/(3+4j)**2])
        assert_almost_equal(x**(-3), [(-11+2j)/125, (-46-9j)/2197,
                                      (-117-44j)/15625])
        assert_almost_equal(x**(0.5), [cmath.sqrt(1+2j), cmath.sqrt(2+3j),
                                       cmath.sqrt(3+4j)])
        norm = 1./((x**14)[0])
        assert_almost_equal(x**14 * norm,
                [i * norm for i in [-76443+16124j, 23161315+58317492j,
                                    5583548873 +  2465133864j]])

        def assert_complex_equal(x, y):
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        for z in [complex(0, np.inf), complex(1, np.inf)]:
            z = blaze.array([z], dshape="complex[float64]")
            assert_complex_equal(z**1, z)
            assert_complex_equal(z**2, z*z)
            assert_complex_equal(z**3, z*z*z)

    def test_power_zero(self):
        zero = blaze.array([0j])
        one = blaze.array([1+0j])
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
    def test_log_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64']:
            log2_ = 0.69314718055994530943
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)*log2_
            result = blaze.log(xf)
            assert_almost_equal(result, yf)


class TestExp(unittest.TestCase):
    def test_exp_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64']:
            log2_ = 0.69314718055994530943
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)*log2_
            result = blaze.exp(yf)
            assert_almost_equal(result, xf)


class TestLogAddExp(unittest.TestCase):
    def test_logaddexp_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for ds, dec in zip(['float32', 'float64'], [6, 15]):
            xf = blaze.log(blaze.array(x, dshape=ds))
            yf = blaze.log(blaze.array(y, dshape=ds))
            zf = blaze.log(blaze.array(z, dshape=ds))
            result = blaze.logaddexp(xf, yf)
            assert_almost_equal(result, zf, decimal=dec)

    def test_logaddexp_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for ds in ['float32', 'float64']:
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp(logxf, logyf)
            assert_almost_equal(result, logzf)

    def test_inf(self):
        inf = blaze.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        for ds in ['float32', 'float64']:
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp(logxf, logyf)
            assert_equal(result, logzf)

    def test_nan(self):
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, blaze.inf)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.inf, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, 0)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(0, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp(blaze.nan, blaze.nan)))


class TestLog2(unittest.TestCase):
    def test_log2_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64']:
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)
            result = blaze.log2(xf)
            assert_almost_equal(result, yf)


class TestLog10(unittest.TestCase):
    def test_log10_values(self):
        x = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64']:
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)
            result = blaze.log10(xf)
            assert_almost_equal(result, yf)


class TestExp2(unittest.TestCase):
    def test_exp2_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for ds in ['float32', 'float64']:
            xf = blaze.array(x, dshape=ds)
            yf = blaze.array(y, dshape=ds)
            result = blaze.exp2(yf)
            assert_almost_equal(result, xf)


class TestLogAddExp2(unittest.TestCase):
    # Need test for intermediate precisions
    def test_logaddexp2_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for ds, dec in zip(['float32', 'float64'], [6, 15, 15]):
            xf = blaze.log2(blaze.array(x, dshape=ds))
            yf = blaze.log2(blaze.array(y, dshape=ds))
            zf = blaze.log2(blaze.array(z, dshape=ds))
            result = blaze.logaddexp2(xf, yf)
            assert_almost_equal(result, zf, decimal=dec)

    def test_logaddexp2_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for ds in ['float32', 'float64']:
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp2(logxf, logyf)
            assert_almost_equal(result, logzf)

    def test_inf(self):
        inf = blaze.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        for ds in ['float32', 'float64']:
            logxf = blaze.array(x, dshape=ds)
            logyf = blaze.array(y, dshape=ds)
            logzf = blaze.array(z, dshape=ds)
            result = blaze.logaddexp2(logxf, logyf)
            assert_equal(result, logzf)

    def test_nan(self):
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, blaze.inf)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.inf, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, 0)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(0, blaze.nan)))
        self.assertTrue(blaze.isnan(blaze.logaddexp2(blaze.nan, blaze.nan)))


class TestRint(unittest.TestCase):
    def test_rint(self):
        a = blaze.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        b = blaze.array([-2., -2., -0.,  0.,  2.,  2.,  2.])
        result = blaze.rint(a)
        assert_equal(result, b)


class TestSign(unittest.TestCase):
    def test_sign(self):
        a = blaze.array([blaze.inf, -blaze.inf, blaze.nan, 0.0, 3.0, -3.0])
        tgt = blaze.array([1., -1., blaze.nan, 0.0, 1.0, -1.0])

        result = blaze.sign(a)
        assert_equal(result, tgt)


class TestExpm1(unittest.TestCase):
    def test_expm1(self):
        assert_almost_equal(blaze.expm1(0.2), blaze.exp(0.2)-1)
        assert_almost_equal(blaze.expm1(1e-6), blaze.exp(1e-6)-1)


class TestLog1p(unittest.TestCase):
    def test_log1p(self):
        assert_almost_equal(blaze.log1p(0.2), blaze.log(1.2))
        assert_almost_equal(blaze.log1p(1e-6), blaze.log(1+1e-6))


class TestSqrt(unittest.TestCase):
    def test_sqrt(self):
        a = blaze.array([0., 9., 64., 1e20, 12345])
        b = blaze.array([0., 3., 8., 1e10, math.sqrt(12345)])
        result = blaze.sqrt(a)
        assert_almost_equal(result, b)


class TestSquare(unittest.TestCase):
    def test_square(self):
        a = blaze.array([0., 3., 8., 1e10, math.sqrt(12345)])
        b = blaze.array([0., 9., 64., 1e20, 12345])
        result = blaze.square(a)
        assert_almost_equal(result, b)
        result = blaze.square(-a)
        assert_almost_equal(result, b)


class TestReciprocal(unittest.TestCase):
    def test_reciprocal(self):
        a = blaze.array([1, 2., 3.33])
        b = blaze.array([1., 0.5, 0.3003003])
        result = blaze.reciprocal(a)
        assert_almost_equal(result, b)


class TestAngles(unittest.TestCase):
    def test_degrees(self):
        assert_almost_equal(blaze.degrees(math.pi), 180.0)
        assert_almost_equal(blaze.degrees(-0.5*math.pi), -90.0)
        assert_almost_equal(blaze.rad2deg(math.pi), 180.0)
        assert_almost_equal(blaze.rad2deg(-0.5*math.pi), -90.0)

    def test_radians(self):
        assert_almost_equal(blaze.radians(180.0), math.pi)
        assert_almost_equal(blaze.radians(-90.0), -0.5*math.pi)
        assert_almost_equal(blaze.deg2rad(180.0), math.pi)
        assert_almost_equal(blaze.deg2rad(-90.0), -0.5*math.pi)


class TestMod(unittest.TestCase):
    def test_remainder_mod_int(self):
        a = blaze.array([-3, -2, -1, 0, 1, 2, 3])
        a_mod_2 = blaze.array([1,  0, 1,  0, 1,  0,  1])
        a_mod_3 = blaze.array([0,  1, 2,  0, 1,  2,  0])
        assert_equal(blaze.remainder(a, 2), a_mod_2)
        assert_equal(blaze.mod(a, 2), a_mod_2)
        assert_equal(blaze.remainder(a, 3), a_mod_3)
        assert_equal(blaze.mod(a, 3), a_mod_3)

    def test_remainder_mod_float(self):
        a = blaze.array([-3, -2, -1, 0, 1, 2, 3], dshape='float32')
        a_mod_2 = blaze.array([1,  0, 1,  0, 1,  0,  1], dshape='float32')
        a_mod_3 = blaze.array([0,  1, 2,  0, 1,  2,  0], dshape='float32')
        assert_equal(blaze.remainder(a, 2), a_mod_2)
        assert_equal(blaze.mod(a, 2), a_mod_2)
        assert_equal(blaze.remainder(a, 3), a_mod_3)
        assert_equal(blaze.mod(a, 3), a_mod_3)

    def test_fmod_int(self):
        a = blaze.array([-3, -2, -1, 0, 1, 2, 3])
        a_fmod_2 = blaze.array([-1,  0, -1,  0, 1,  0,  1])
        a_fmod_3 = blaze.array([0,  -2, -1,  0, 1,  2,  0])
        assert_equal(blaze.fmod(a, 2), a_fmod_2)
        assert_equal(blaze.fmod(a, 3), a_fmod_3)

    def test_fmod_float(self):
        a = blaze.array([-3, -2, -1, 0, 1, 2, 3], dshape='float32')
        a_fmod_2 = blaze.array([-1,  0, -1,  0, 1,  0,  1], dshape='float32')
        a_fmod_3 = blaze.array([0,  -2, -1,  0, 1,  2,  0], dshape='float32')
        assert_equal(blaze.fmod(a, 2), a_fmod_2)
        assert_equal(blaze.fmod(a, 3), a_fmod_3)


class TestAbs(unittest.TestCase):
    def test_simple(self):
        x = blaze.array([1+1j, 0+2j, 1+2j, blaze.inf, blaze.nan])
        y_r = blaze.array([blaze.sqrt(2.), 2, blaze.sqrt(5),
                           blaze.inf, blaze.nan])
        y = blaze.abs(x)
        for i in range(len(x)):
            assert_almost_equal(y[i], y_r[i])

    def test_fabs(self):
        # Test that blaze.abs(x +- 0j) == blaze.abs(x)
        # (as mandated by C99 for cabs)
        x = blaze.array([1+0j], dshape="complex[float64]")
        assert_array_equal(blaze.abs(x), blaze.real(x))

        x = blaze.array([complex(1, -0.)], dshape="complex[float64]")
        assert_array_equal(blaze.abs(x), blaze.real(x))

        x = blaze.array([complex(blaze.inf, -0.)], dshape="complex[float64]")
        assert_array_equal(blaze.abs(x), blaze.real(x))

        x = blaze.array([complex(blaze.nan, -0.)], dshape="complex[float64]")
        assert_array_equal(blaze.abs(x), blaze.real(x))

    def test_cabs_inf_nan(self):
        # cabs(+-nan + nani) returns nan
        self.assertTrue(blaze.isnan(blaze.abs(complex(blaze.nan, blaze.nan))))
        self.assertTrue(blaze.isnan(blaze.abs(complex(-blaze.nan, blaze.nan))))
        self.assertTrue(blaze.isnan(blaze.abs(complex(blaze.nan, -blaze.nan))))
        self.assertTrue(blaze.isnan(blaze.abs(complex(-blaze.nan, -blaze.nan))))

        # According to C99 standard, if exactly one of the real/part is inf and
        # the other nan, then cabs should return inf
        assert_equal(blaze.abs(complex(blaze.inf, blaze.nan)), blaze.inf)
        assert_equal(blaze.abs(complex(blaze.nan, blaze.inf)), blaze.inf)
        assert_equal(blaze.abs(complex(-blaze.inf, blaze.nan)), blaze.inf)
        assert_equal(blaze.abs(complex(blaze.nan, -blaze.inf)), blaze.inf)

        values = [complex(blaze.nan, blaze.nan),
                  complex(-blaze.nan, blaze.nan),
                  complex(blaze.inf, blaze.nan),
                  complex(-blaze.inf, blaze.nan)]

        for z in values:
            abs_conj_z = blaze.abs(blaze.conj(z))
            conj_abs_z = blaze.conj(blaze.abs(z))
            abs_z = blaze.abs(z)
            assert_equal(abs_conj_z, conj_abs_z)
            assert_equal(abs_conj_z, abs_z)
            assert_equal(conj_abs_z, abs_z)


class TestTrig(unittest.TestCase):
    def test_sin(self):
        a = blaze.array([0, math.pi/6, math.pi/3, 0.5*math.pi,
                         math.pi, 1.5*math.pi, 2*math.pi])
        b = blaze.array([0, 0.5, 0.5*blaze.sqrt(3), 1, 0, -1, 0])
        assert_allclose(blaze.sin(a), b, rtol=1e-15, atol=1e-15)
        assert_allclose(blaze.sin(-a), -b, rtol=1e-15, atol=1e-15)

    def test_cos(self):
        a = blaze.array([0, math.pi/6, math.pi/3, 0.5*math.pi,
                         math.pi, 1.5*math.pi, 2*math.pi])
        b = blaze.array([1, 0.5*blaze.sqrt(3), 0.5, 0, -1, 0, 1])
        assert_allclose(blaze.cos(a), b, rtol=1e-15, atol=1e-15)
        assert_allclose(blaze.cos(-a), b, rtol=1e-15, atol=1e-15)


def _check_branch_cut(f, x0, dx, re_sign=1, im_sign=-1, sig_zero_ok=False,
                      dtype=np.complex):
    """
    Check for a branch cut in a function.

    Assert that `x0` lies on a branch cut of function `f` and `f` is
    continuous from the direction `dx`.

    Parameters
    ----------
    f : func
        Function to check
    x0 : array-like
        Point on branch cut
    dx : array-like
        Direction to check continuity in
    re_sign, im_sign : {1, -1}
        Change of sign of the real or imaginary part expected
    sig_zero_ok : bool
        Whether to check if the branch cut respects signed zero (if applicable)
    dtype : dtype
        Dtype to check (should be complex)

    """
    x0 = np.atleast_1d(x0).astype(dtype)
    dx = np.atleast_1d(dx).astype(dtype)

    scale = np.finfo(dtype).eps * 1e3
    atol  = 1e-4

    y0 = f(x0)
    yp = f(x0 + dx*scale*np.absolute(x0)/np.absolute(dx))
    ym = f(x0 - dx*scale*np.absolute(x0)/np.absolute(dx))

    assert_(np.all(np.absolute(y0.real - yp.real) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.imag - yp.imag) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.real - ym.real*re_sign) < atol), (y0, ym))
    assert_(np.all(np.absolute(y0.imag - ym.imag*im_sign) < atol), (y0, ym))

    if sig_zero_ok:
        # check that signed zeros also work as a displacement
        jr = (x0.real == 0) & (dx.real != 0)
        ji = (x0.imag == 0) & (dx.imag != 0)

        x = -x0
        x.real[jr] = 0.*dx.real
        x.imag[ji] = 0.*dx.imag
        x = -x
        ym = f(x)
        ym = ym[jr | ji]
        y0 = y0[jr | ji]
        assert_(np.all(np.absolute(y0.real - ym.real*re_sign) < atol), (y0, ym))
        assert_(np.all(np.absolute(y0.imag - ym.imag*im_sign) < atol), (y0, ym))


class TestComplexFunctions(unittest.TestCase):
    funcs = [blaze.arcsin, blaze.arccos,  blaze.arctan, blaze.arcsinh,
             blaze.arccosh, blaze.arctanh, blaze.sin, blaze.cos, blaze.tan,
             blaze.exp, blaze.exp2, blaze.log, blaze.sqrt, blaze.log10,
             blaze.log2, blaze.log1p]

    def test_it(self):
        for f in self.funcs:
            if f is blaze.arccosh:
                x = 1.5
            else:
                x = .5
            fr = f(x)
            fz = f(complex(x))
            assert_almost_equal(fz.real, fr, err_msg='real part %s' % f)
            assert_almost_equal(fz.imag, 0., err_msg='imag part %s' % f)

    def test_precisions_consistent(self):
        z = 1 + 1j
        for f in self.funcs:
            fcf = f(blaze.array(z, dshape='complex[float32]'))
            fcd = f(blaze.array(z, dshape='complex[float64]'))
            assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s' % f)

    def test_branch_cuts(self):
        # check branch cuts and continuity on them
        _check_branch_cut(blaze.log,   -0.5, 1j, 1, -1)
        _check_branch_cut(blaze.log2,  -0.5, 1j, 1, -1)
        _check_branch_cut(blaze.log10, -0.5, 1j, 1, -1)
        _check_branch_cut(blaze.log1p, -1.5, 1j, 1, -1)
        _check_branch_cut(blaze.sqrt,  -0.5, 1j, 1, -1)

        _check_branch_cut(blaze.arcsin, [-2, 2],   [1j, -1j], 1, -1)
        _check_branch_cut(blaze.arccos, [-2, 2],   [1j, -1j], 1, -1)
        _check_branch_cut(blaze.arctan, [-2j, 2j],  [1,  -1], -1, 1)

        _check_branch_cut(blaze.arcsinh, [-2j,  2j], [-1,   1], -1, 1)
        _check_branch_cut(blaze.arccosh, [-1, 0.5], [1j,  1j], 1, -1)
        _check_branch_cut(blaze.arctanh, [-2,   2], [1j, -1j], 1, -1)

        # check against bogus branch cuts: assert continuity between quadrants
        _check_branch_cut(blaze.arcsin, [-2j, 2j], [1,  1], 1, 1)
        _check_branch_cut(blaze.arccos, [-2j, 2j], [1,  1], 1, 1)
        _check_branch_cut(blaze.arctan, [-2,  2], [1j, 1j], 1, 1)

        _check_branch_cut(blaze.arcsinh, [-2,  2, 0], [1j, 1j, 1], 1, 1)
        _check_branch_cut(blaze.arccosh, [-2j, 2j, 2], [1,  1,  1j], 1, 1)
        _check_branch_cut(blaze.arctanh, [-2j, 2j, 0], [1,  1,  1j], 1, 1)

    @skip("These branch cuts are known to fail")
    def test_branch_cuts_failing(self):
        # XXX: signed zero not OK with ICC on 64-bit platform for log, see
        # http://permalink.gmane.org/gmane.comp.python.numeric.general/25335
        _check_branch_cut(blaze.log,   -0.5, 1j, 1, -1, True)
        _check_branch_cut(blaze.log2,  -0.5, 1j, 1, -1, True)
        _check_branch_cut(blaze.log10, -0.5, 1j, 1, -1, True)
        _check_branch_cut(blaze.log1p, -1.5, 1j, 1, -1, True)
        # XXX: signed zeros are not OK for sqrt or for the arc* functions
        _check_branch_cut(blaze.sqrt,  -0.5, 1j, 1, -1, True)
        _check_branch_cut(blaze.arcsin, [-2, 2],   [1j, -1j], 1, -1, True)
        _check_branch_cut(blaze.arccos, [-2, 2],   [1j, -1j], 1, -1, True)
        _check_branch_cut(blaze.arctan, [-2j, 2j],  [1,  -1], -1, 1, True)
        _check_branch_cut(blaze.arcsinh, [-2j,  2j], [-1,   1], -1, 1, True)
        _check_branch_cut(blaze.arccosh, [-1, 0.5], [1j,  1j], 1, -1, True)
        _check_branch_cut(blaze.arctanh, [-2,   2], [1j, -1j], 1, -1, True)

    def test_against_cmath(self):
        import cmath

        points = [-1-1j, -1+1j, +1-1j, +1+1j]
        name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
                    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
        atol = 4*np.finfo(np.complex).eps
        for func in self.funcs:
            fname = func.name
            cname = name_map.get(fname, fname)
            try:
                cfunc = getattr(cmath, cname)
            except AttributeError:
                continue
            for p in points:
                a = complex(func(complex(p)))
                b = cfunc(p)
                self.assertTrue(abs(a - b) < atol,
                                "%s %s: %s; cmath: %s" % (fname, p, a, b))


class TestMaximum(unittest.TestCase):
    def test_float_nans(self):
        nan = blaze.nan
        arg1 = blaze.array([0, nan, nan])
        arg2 = blaze.array([nan, 0, nan])
        out = blaze.array([nan, nan, nan])
        assert_equal(blaze.maximum(arg1, arg2), out)

    def test_complex_nans(self):
        nan = blaze.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = blaze.array([0, cnan, cnan])
            arg2 = blaze.array([cnan, 0, cnan])
            out = blaze.array([nan, nan, nan],
                              dshape=datashape.complex_float64)
            assert_equal(blaze.maximum(arg1, arg2), out)


class TestMinimum(unittest.TestCase):
    def test_float_nans(self):
        nan = blaze.nan
        arg1 = blaze.array([0,   nan, nan])
        arg2 = blaze.array([nan, 0,   nan])
        out = blaze.array([nan, nan, nan])
        assert_equal(blaze.minimum(arg1, arg2), out)

    def test_complex_nans(self):
        nan = blaze.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = blaze.array([0, cnan, cnan])
            arg2 = blaze.array([cnan, 0, cnan])
            out = blaze.array([nan, nan, nan],
                              dshape=datashape.complex_float64)
            assert_equal(blaze.minimum(arg1, arg2), out)


class TestFmax(unittest.TestCase):
    def test_float_nans(self):
        nan = blaze.nan
        arg1 = blaze.array([0, nan, nan])
        arg2 = blaze.array([nan, 0, nan])
        out = blaze.array([0, 0, nan])
        assert_equal(blaze.fmax(arg1, arg2), out)

    def test_complex_nans(self):
        nan = blaze.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = blaze.array([0, cnan, cnan])
            arg2 = blaze.array([cnan, 0, cnan])
            out = blaze.array([0, 0, nan],
                              dshape=datashape.complex_float64)
            assert_equal(blaze.fmax(arg1, arg2), out)


class TestFmin(unittest.TestCase):
    def test_float_nans(self):
        nan = blaze.nan
        arg1 = blaze.array([0, nan, nan])
        arg2 = blaze.array([nan, 0, nan])
        out = blaze.array([0, 0, nan])
        assert_equal(blaze.fmin(arg1, arg2), out)

    def test_complex_nans(self):
        nan = blaze.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = blaze.array([0, cnan, cnan])
            arg2 = blaze.array([cnan, 0, cnan])
            out = blaze.array([0, 0, nan], dshape=datashape.complex_float64)
            assert_equal(blaze.fmin(arg1, arg2), out)


if __name__ == '__main__':
    unittest.main()
