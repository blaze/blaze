# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

import blaze
from blaze import error
from blaze.ops.ufuncs import add, mul
from blaze.air import interps
from blaze.datashape import unify, dshapes, coerce, normalize

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestPython(unittest.TestCase):

    def test_interp(self):
        a = blaze.array(range(10))
        b = blaze.array([float(x) for x in range(10)])
        result = add(a, b)

if __name__ == '__main__':
    # TestPython('test_interp').debug()
    unittest.main()