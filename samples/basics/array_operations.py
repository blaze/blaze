'''Sample script showing off array basic operations'''

from __future__ import absolute_import, division, print_function
from random import randint

import blaze


def make_test_array(datashape):
    return blaze.ones(datashape) * randint(1, 10)


def test_operations(datashape):
    a = make_test_array(datashape)
    b = make_test_array(datashape)
    print('a:\n', a)
    print('b:\n', b)
    print('a + b:\n', a + b)
    print('a - b:\n', a - b)
    print('a * b:\n', a * b)
    print('a / b:\n', a / b)
    print('blaze.max(a):\n', blaze.max(a))
    print('blaze.min(a):\n', blaze.min(a))
    print('blaze.product(a):\n', blaze.product(a))
    print('blaze.sum(a):\n', blaze.sum(a))


if __name__ == '__main__':
    test_operations('10 * float32')
    test_operations('10 * int32')
    test_operations('10 * 10 * float64')
