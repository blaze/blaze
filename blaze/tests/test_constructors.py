"""Test constructors

Test constructors for arrays of basic types.
"""


from blaze import zeros, ones, dshape

from numpy.testing.decorators import skipif
from unittest import TestCase


class ConstructTest(TestCase):
    full_types = [
        '10, int8',
        '30, int16',
        '40, int32',
        '12, int64',
        '10, uint8',
        '30, uint16',
        '40, uint32',
        '12, uint64',
        '10, float16',
        '15, float32',
        '32, float64',
        '20, complex64',
        '30, complex128',
        '10, 20, int32',
        '10, 20, uint8',
        '10, 20, float32',
        '10, 20, complex128',
        ]

    def setUp(self):
        import sys
        if sys.platform != 'win32':
            full_types.extend([
                '32, float128',
                '30, complex256'
                ])

    def test_zeros(self):
        for t in self.full_types:
            a = zeros(t)
            self.assertEqual(a.datashape, dshape(t))

    def test_ones(self):
        for t in self.full_types:
            a = ones(t)
            self.assertEqual(a.datashape, dshape(t))



## Local Variables:
## mode: python
## coding: utf-8 
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:
