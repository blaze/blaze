"""Utilities for the high level Blaze test suite"""

from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os
import shutil
import glob


# Useful superclass for disk-based tests
class PersistentTest():

    def setUp(self):
        if self.pformat == "blz":
            prefix = 'barray-' + self.__class__.__name__
            self.root = tempfile.mkdtemp(prefix=prefix)
            os.rmdir(self.root)

        elif self.pformat == "hdf5":
            prefix = 'hdf5-' + self.__class__.__name__
            _, self.root = tempfile.mkstemp(suffix='.h5', prefix=prefix)
            os.remove(self.root)

    def tearDown(self):
        if self.pformat == "blz":
            for dir_ in glob.glob(self.root+'*'):
                shutil.rmtree(dir_)
        elif self.pformat == "hdf5":
            os.remove(self.root)


class BTestCase(unittest.TestCase):
    """
    TestCase that provides some stuff missing in 2.6.
    """

    def assertIsInstance(self, obj, cls, msg=None):
        self.assertTrue(isinstance(obj, cls),
                        msg or "%s is not an instance of %s" % (obj, cls))

    def assertGreater(self, a, b, msg=None):
        self.assertTrue(a > b, msg or "%s is not greater than %s" % (a, b))

    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg or "%s is not greater than %s" % (a, b))
