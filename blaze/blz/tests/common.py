import unittest
import tempfile
import os, os.path
import glob
import shutil
from blaze import blz
import numpy as np

# Global variables for the tests
verbose = False
heavy = False



# Useful superclass for disk-based tests
class MayBeDiskTest():

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


