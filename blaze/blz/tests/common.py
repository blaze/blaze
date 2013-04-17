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



#
# The original constructors of barray are gone in blaze
# monkey-patch them in order to retain the code in the tests
#
def _barray_fill(shape, value, dtype = None, cparams = None, rootdir = None):
    """
    hacked version that will only work with 2 dimensions, as needed
    by the tests
    """
    arr = np.empty(shape, dtype=dtype)
    arr[:,:]=value
    carr = blz.barray(arr, cparams=cparams, rootdir=rootdir) 
    return carr

blz.fill = _barray_fill

def _barray_arange(*args, **kw_args):
    array = np.arange(*args)
    return blz.barray(array, **kw_args)

blz.arange = _barray_arange
#
# end monkey-patch
#




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


