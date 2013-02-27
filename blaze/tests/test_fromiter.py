"""Test fromiter blaze constructors

Test on both, memory arrays and persistent arrays
"""

from blaze import fromiter, params, open, dshape, to_numpy
from numpy.testing.decorators import skipif
from unittest import TestCase

def blaze_remove_persistent_table(table_name):
    from shutils import rmtree
    import os

    if os.path.exists(table_name):
        rmtree(table_name)
    else:
        raise Exception('table %s does not exist.' % table_name)


class FromiterTemplate:
    @skipif(True)
    def test_array(self):
        t = fromiter(self.gen(), self.ds, self.p)
        shape, dtype = to_numpy(t.datashape)
        shape_orig, dtype_orig = to_numpy(self.ds)
        self.assertEqual(dtype, dtype_orig)
        self.assertEqual(len(shape), 1)
        self.assertEqual(shape[0], self.count)


class FromIterMemory_floatarray(FromiterTemplate, TestCase):
    ds = dshape('(x, float32)')
    count = 1000
    p = params(clevel=5)

    def gen(self):
        return (i for i in xrange(self.count))


class FromIterMemory_doublearray(FromiterTemplate, TestCase):
    ds = dshape('(x, float64)')
    count = 1000
    p = params(clevel=5)

    def gen(self):
        return (i for i in xrange(self.count))


class FromIterMemory_int32array(FromiterTemplate, TestCase):
    ds = dshape('(x, int32)')
    count = 1000
    p = params(clevel=5)

    def gen(self):
        return (i for i in xrange(self.count))


class FromIterMemory_int64array(FromiterTemplate, TestCase): 
    ds = dshape('(x, int64)')
    count = 1000
    p = params(clevel=5)

    def gen(self):
        return (i for i in xrange(self.count))


## Local Variables:
## mode: python
## coding: utf-8 
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:
