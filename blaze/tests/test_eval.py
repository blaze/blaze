from __future__ import absolute_import, division, print_function

import os
import sys
import unittest
import tempfile
from itertools import product as it_product

import blaze
from blaze.datadescriptor import ddesc_as_py


import numpy as np
from numpy.testing import assert_allclose

def _clean_disk_arrays():
    try:
        from shutil import rmtree
        rmtree(tmpdir)
    except Exception as e:
        print('Error cleaning up temp dir %s:\n%s' % (tmpdir, e))

def _mk_dir():
    global tmpdir
    tmpdir = tempfile.mkdtemp(prefix='blztmp')

def _ddesc(name):
    path = os.path.join(tmpdir, name + '.blz')
    return blaze.BLZ_DDesc(path, mode='w')

def _addition(a,b):
    return (a+b)
def _expression(a, b):
    return (a+b)*(a+b)

#------------------------------------------------------------------------
# Test Generation
#------------------------------------------------------------------------

def _add_tests():
    _pair = ['mem', 'dsk']
    frame = sys._getframe(1)
    for expr, ltr in zip([_addition, _expression], ['R', 'Q']):
        for i in it_product(_pair, _pair, _pair):
            args = i + (ltr,)
            f = _build_tst(expr, *args)
            f.__name__ = 'test_{1}_{2}_to_{3}{0}'.format(f.__name__, *args)
            frame.f_locals[f.__name__] = f

def _build_tst(kernel, storage1, storage2, storage3, R):
    def function(self):
        A = getattr(self, storage1 + 'A')
        B = getattr(self, storage2 + 'B')

        Rd = kernel(A, B)
        self.assert_(isinstance(Rd, blaze.Array))
        self.assert_(Rd.ddesc.capabilities.deferred)
        p = _ddesc(storage3 + 'Rd') if storage3 == 'dsk' else None
        try:
            Rc = blaze.eval(Rd, ddesc=p)
            self.assert_(isinstance(Rc, blaze.Array))
            npy_data = getattr(self, 'npy' + R)
            assert_allclose(np.array(ddesc_as_py(Rc.ddesc)), npy_data)

            if storage3 == 'dsk':
                self.assert_(Rc.ddesc.capabilities.persistent)
            else:
                self.assert_(not Rc.ddesc.capabilities.persistent)

        finally:
            try:
                if p is not None:
                    blaze.drop(p)
            except:
                pass # show the real error...

    return function


#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestEvalScalar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npyA = np.array(10)
        cls.npyB = np.arange(0.0, 100.0)
        cls.npyR = _addition(cls.npyA, cls.npyB)
        cls.npyQ = _expression(cls.npyA, cls.npyB)
        cls.memA = blaze.array(cls.npyA)
        cls.memB = blaze.array(cls.npyB)

        _mk_dir()
        cls.dskA = blaze.array(cls.npyA, ddesc=_ddesc('dskA'))
        cls.dskB = blaze.array(cls.npyB, ddesc=_ddesc('dskB'))

    @classmethod
    def tearDownClass(cls):
        _clean_disk_arrays()
        del(cls.npyA)
        del(cls.npyB)
        del(cls.npyR)
        del(cls.memA)
        del(cls.memB)
        del(cls.dskA)
        del(cls.dskB)

    # add all tests for all permutations
    # TODO: Enable. Currently segfaults
    # _add_tests()


class TestEval1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npyA = np.arange(0.0, 100.0)
        cls.npyB = np.arange(0.0, 100.0)
        cls.npyR = _addition(cls.npyA, cls.npyB)
        cls.npyQ = _expression(cls.npyA, cls.npyB)
        cls.memA = blaze.array(cls.npyA)
        cls.memB = blaze.array(cls.npyB)

        _mk_dir()
        cls.dskA = blaze.array(cls.npyA, ddesc=_ddesc('dskA'))
        cls.dskB = blaze.array(cls.npyB, ddesc=_ddesc('dskB'))

    @classmethod
    def tearDownClass(cls):
        _clean_disk_arrays()
        del(cls.npyA)
        del(cls.npyB)
        del(cls.npyR)
        del(cls.memA)
        del(cls.memB)
        del(cls.dskA)
        del(cls.dskB)

    # add all tests for all permutations
    _add_tests()


class TestEval2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npyA = np.arange(0.0, 100.0).reshape(20, 5)
        cls.npyB = np.arange(0.0, 100.0).reshape(20, 5)
        cls.npyR = _addition(cls.npyA, cls.npyB)
        cls.npyQ = _expression(cls.npyA, cls.npyB)
        cls.memA = blaze.array(cls.npyA)
        cls.memB = blaze.array(cls.npyB)

        _mk_dir()
        cls.dskA = blaze.array(cls.npyA, ddesc=_ddesc('dskA'))
        cls.dskB = blaze.array(cls.npyB, ddesc=_ddesc('dskB'))

    @classmethod
    def tearDownClass(cls):
        _clean_disk_arrays()
        del(cls.npyA)
        del(cls.npyB)
        del(cls.npyR)
        del(cls.memA)
        del(cls.memB)
        del(cls.dskA)
        del(cls.dskB)

    # add all tests for all permutations
    _add_tests()

if __name__ == '__main__':
    #TestEval2D.setUpClass()
    #TestEval2D('test_dsk_mem_to_memfunction').debug()
    unittest.main(verbosity=2)

