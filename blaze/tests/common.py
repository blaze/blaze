"""Utilities for the high level Blaze test suite"""

from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os
import shutil
import glob


# Useful superclass for disk-based tests
class MayBePersistentTest():

    disk = False

    def setUp(self):
        if self.disk:
            prefix = 'barray-' + self.__class__.__name__
            self.rootdir = tempfile.mkdtemp(prefix=prefix)
            os.rmdir(self.rootdir)  # tests needs this cleared
        else:
            self.rootdir = None

    def tearDown(self):
        if self.disk:
            # Remove every directory starting with rootdir
            for dir_ in glob.glob(self.rootdir+'*'):
                shutil.rmtree(dir_)


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
