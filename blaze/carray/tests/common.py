import unittest
import tempfile
import os, os.path
import glob
import shutil
import blaze.carray as ca
import numpy as np

# Global variables for the tests
verbose = False
heavy = False



#
# The original constructors of carray are gone in blaze
# monkey-patch them in order to retain the code in the tests
#
def _carray_fill(shape, value, dtype = None, cparams = None, rootdir = None):
    """
    hacked version that will only work with 2 dimensions, as needed
    by the tests
    """
    arr = np.empty(shape, dtype=dtype)
    arr[:,:]=value
    carr = ca.carray(arr, cparams=cparams, rootdir=rootdir) 
    return carr

ca.fill = _carray_fill

def _carray_arange(*args, **kw_args):
    array = np.arange(*args)
    return ca.carray(array, **kw_args)

ca.arange = _carray_arange
#
# end monkey-patch
#




# Useful superclass for disk-based tests
class MayBeDiskTest():

    disk = False

    def setUp(self):
        if self.disk:
            prefix = 'carray-' + self.__class__.__name__
            self.rootdir = tempfile.mkdtemp(prefix=prefix)
            os.rmdir(self.rootdir)  # tests needs this cleared
        else:
            self.rootdir = None

    def tearDown(self):
        if self.disk:
            # Remove every directory starting with rootdir
            for dir_ in glob.glob(self.rootdir+'*'):
                shutil.rmtree(dir_)


