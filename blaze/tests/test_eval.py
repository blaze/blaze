from __future__ import (print_function, absolute_import)

import blaze

import os
import sys
import numpy as np
from numpy.testing import assert_allclose
import unittest
import tempfile
from itertools import product as it_product
from blaze.datadescriptor import dd_as_py
from blaze.py3help import exec_

def _clean_disk_arrays():
    try:
        from shutil import rmtree
        rmtree(tmpdir)
    except Exception as e:
        print('Error cleaning up temp dir %s:\n%s' % (tmpdir, e))

def _mk_dir():
    global tmpdir
    tmpdir = tempfile.mkdtemp(prefix='blztmp')

def _persist(name):
    return blaze.Storage(os.path.join(tmpdir, name + '.blz'))

def _addition(a,b):
    return (a+b)
def _expression(a, b):
    return (a+b)*(a+b)

def _add_tests():
    _pair = ['mem', 'dsk']

    _template = '''
def test_{1}_{2}_to_{3}{0}(self):
    Rd = {0}(self.{1}A, self.{2}B)
    self.assert_(isinstance(Rd, blaze.Array))
    self.assert_(Rd._data.deferred)
    p = _persist('{3}Rd') if '{3}' == 'dsk' else None
    try:
        Rc = blaze.eval(Rd, persist=p)
        self.assert_(isinstance(Rc, blaze.Array))
        assert_allclose(np.array(dd_as_py(Rc._data)), self.npy{4})
        self.assert_(Rc._data.persistent if '{3}' == 'dsk'
                                         else not Rc._data.persistent)
    finally:
        if p is not None:
            blaze.drop(p)
'''
    frame = sys._getframe(1)
    for expr, ltr in zip(['_addition', '_expression'], ['R', 'Q']):
        for i in it_product(_pair, _pair, _pair):
            args = i + (ltr,)
            exec_(_template.format(expr,*args),
                  frame.f_globals,
                  frame.f_locals)



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
        cls.dskA = blaze.array(cls.npyA, persist=_persist('dskA'))
        cls.dskB = blaze.array(cls.npyB, persist=_persist('dskB'))

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
        cls.dskA = blaze.array(cls.npyA, persist=_persist('dskA'))
        cls.dskB = blaze.array(cls.npyB, persist=_persist('dskB'))

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
    unittest.main(verbosity=2)

