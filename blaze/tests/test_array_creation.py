from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
import datashape
import blaze
from blaze.datadescriptor import ddesc_as_py
from blaze.tests.common import MayBePersistentTest
from blaze import (append,
    DyND_DDesc, BLZ_DDesc, HDF5_DDesc)


from blaze.py2help import skip, skipIf
import blz

from blaze.optional_packages import tables_is_here
if tables_is_here:
    import tables as tb


class TestEphemeral(unittest.TestCase):
    def test_create_scalar(self):
        a = blaze.array(True)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('bool'))
        self.assertEqual(bool(a), True)
        a = blaze.array(-123456)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('int32'))
        self.assertEqual(int(a), -123456)
        a = blaze.array(-1.25e-10)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('float64'))
        self.assertEqual(float(a), -1.25e-10)
        a = blaze.array(-1.25e-10+2.5j)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('complex[float64]'))
        self.assertEqual(complex(a), -1.25e-10+2.5j)

    def test_create_from_numpy(self):
        a = blaze.array(np.arange(3))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [0, 1, 2])

    def test_create(self):
        # A default array (backed by DyND)
        a = blaze.array([1,2,3])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertTrue(str(a.dshape) == "3 * int32")
        self.assertEqual(ddesc_as_py(a.ddesc), [1, 2, 3])

    def test_create_dshape(self):
        # A default array (backed by DyND)
        a = blaze.array([1,2,3], 'float64')
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertTrue(str(a.dshape) == "3 * float64")
        self.assertEqual(ddesc_as_py(a.ddesc), [1, 2, 3])

    def test_create_append(self):
        # A default array (backed by DyND, append not supported yet)
        a = blaze.array([])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertRaises(ValueError, append, a, [1,2,3])

    def test_create_compress(self):
        # A compressed array (backed by BLZ)
        ddesc = BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
        a = blaze.array(np.arange(1,4), ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [1, 2, 3])

    def test_create_iter(self):
        # A simple 1D array
        a = blaze.array(i for i in range(10))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('10 * int32'))
        self.assertEqual(ddesc_as_py(a.ddesc), list(range(10)))
        # A nested iter
        a = blaze.array((i for i in range(x)) for x in range(5))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('5 * var * int32'))
        self.assertEqual(ddesc_as_py(a.ddesc),
                         [[i for i in range(x)] for x in range(5)])
        # A list of iter
        a = blaze.array([range(3), (1.5*x for x in range(4)), iter([-1, 1])])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('3 * var * float64'))
        self.assertEqual(ddesc_as_py(a.ddesc),
                         [list(range(3)),
                          [1.5*x for x in range(4)],
                          [-1, 1]])

    def test_create_compress_iter(self):
        # A compressed array (backed by BLZ)
        ddesc = BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
        a = blaze.array((i for i in range(10)), ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), list(range(10)))

    def test_create_zeros(self):
        # A default array
        a = blaze.zeros('10 * int64')
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [0]*10)

    def test_create_compress_zeros(self):
        # A compressed array (backed by BLZ)
        ddesc = BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
        a = blaze.zeros('10 * int64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [0]*10)

    def test_create_ones(self):
        # A default array
        a = blaze.ones('10 * int64')
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [1]*10)

    def test_create_compress_ones(self):
        # A compressed array (backed by BLZ)
        ddesc = BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
        a = blaze.ones('10 * int64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(ddesc_as_py(a.ddesc), [1]*10)

    def test_create_record(self):
        # A simple record array
        a = blaze.array([(10, 3.5), (15, 2.25)],
                        dshape="var * {val: int32, flt: float32}")
        self.assertEqual(ddesc_as_py(a.ddesc), [{'val': 10, 'flt': 3.5},
                        {'val': 15, 'flt': 2.25}])
        # Test field access via attributes
        aval = a.val
        self.assertEqual(ddesc_as_py(aval.ddesc), [10, 15])
        aflt = a.flt
        self.assertEqual(ddesc_as_py(aflt.ddesc), [3.5, 2.25])


class TestBLZPersistent(MayBePersistentTest, unittest.TestCase):

    disk = True
    dir_ = True

    def test_create(self):
        ddesc = BLZ_DDesc(path=self.rootdir, mode='w')
        a = blaze.array([2], 'float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertTrue(a.dshape.shape == (1,))
        self.assertEqual(ddesc_as_py(a.ddesc), [2])

    def test_append(self):
        ddesc = BLZ_DDesc(path=self.rootdir, mode='w')
        a = blaze.zeros('0 * float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        append(a, list(range(10)))
        self.assertEqual(ddesc_as_py(a.ddesc), list(range(10)))

    # Using a 1-dim as the internal dimension
    def test_append2(self):
        ddesc = BLZ_DDesc(path=self.rootdir, mode='w')
        a = blaze.empty('0 * 2 * float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        lvals = [[i,i*2] for i in range(10)]
        append(a, lvals)
        self.assertEqual(ddesc_as_py(a.ddesc), lvals)


class TestHDF5Persistent(MayBePersistentTest, unittest.TestCase):

    disk = True

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_create(self):
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='w')
        a = blaze.array([2], 'float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertTrue(a.dshape.shape == (1,))
        self.assertEqual(ddesc_as_py(a.ddesc), [2])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_append(self):
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='a')
        a = blaze.zeros('0 * float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        append(a, list(range(10)))
        self.assertEqual(ddesc_as_py(a.ddesc), list(range(10)))

    # Using a 1-dim as the internal dimension
    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_append2(self):
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='a')
        a = blaze.empty('0 * 2 * float64', ddesc=ddesc)
        self.assertTrue(isinstance(a, blaze.Array))
        lvals = [[i,i*2] for i in range(10)]
        append(a, lvals)
        self.assertEqual(ddesc_as_py(a.ddesc), lvals)



if __name__ == '__main__':
    unittest.main(verbosity=2)
