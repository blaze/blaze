# XXX To Be Done: For the time being the tests here are skipped,
# but they should be executed as soon as we implement a way to
# separate execution of normal and heavy tests (via flag to the
# test runner, for example).

import sys
from unittest import TestCase
if sys.version_info >= (2, 7):
    from unittest import skip
else:
    from nose.plugins.skip import SkipTest
    class skip(object):
        def __init__(self, reason):
            self.reason = reason

        def __call__(self, func):
            from nose.plugins.skip import SkipTest
            def wrapped(*args, **kwargs):
                raise SkipTest("Test %s is skipped because: %s" % (func.__name__, self.reason))
            wrapped.__name__ = func.__name__
            return wrapped

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from blaze import blz
from blaze.blz.tests import common
from common import MayBeDiskTest


class largeBarrayTest(MayBeDiskTest, TestCase):

    disk = True

    @skip('')
    def test00(self):
        """Creating an extremely large barray (> 2**32) in memory."""

        cn = blz.zeros(5e9, dtype="i1")
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

    @skip('')
    def test01(self):
        """Creating an extremely large barray (> 2**32) on disk."""

        cn = blz.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

    @skip('')
    def test02(self):
        """Opening an extremely large barray (> 2**32) on disk."""

        # Create the array on-disk
        cn = blz.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))
        # Reopen it from disk
        cn = blz.barray(rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

## Local Variables:
## mode: python
## coding: utf-8
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:

