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
        '10, float16',
        '15, float32',
        '32, float64',
        '10, complex32',
        '20, complex64',
        '30, complex128',
        '10, 20, int32',
        '32, 14, 45, 56, complex64'
        ]

    @skipif(True)
    def test_zeros(self):
        for t in self.full_types:
            a = zeros(t)
            if a.datashape != dshape(t):
                print 'test_zeros fails for \'%s\'' % t



## Local Variables:
## mode: python
## coding: utf-8 
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:
