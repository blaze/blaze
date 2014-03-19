from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os, os.path
import glob
import shutil
import blaze


def remove_tree(rootdir):
    # Remove every directory starting with rootdir
    for dir_ in glob.glob(rootdir+'*'):
        shutil.rmtree(dir_)

# Useful superclass for disk-based tests
class MayBeDiskTest(unittest.TestCase):

    disk = False

    def setUp(self):
        if self.disk:
            prefix = 'blaze-' + self.__class__.__name__
            suffix = '.blz'
            self.rootdir1 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(self.rootdir1)
            self.store1 = blaze.Storage(self.rootdir1)
            self.rootdir2 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(self.rootdir2)
            self.store2 = blaze.Storage(self.rootdir2)
            self.rootdir3 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(self.rootdir3)
            self.store3 = blaze.Storage(self.rootdir3)
        else:
            self.rootdir1 = None
            self.store1 = None
            self.rootdir2 = None
            self.store2 = None
            self.rootdir3 = None
            self.store3 = None

    def tearDown(self):
        if self.disk:
            remove_tree(self.rootdir1)
            remove_tree(self.rootdir2)
            remove_tree(self.rootdir3)
