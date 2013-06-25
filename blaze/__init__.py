from __future__ import absolute_import

# build the blaze namespace with selected functions

from . import datashape, ckernel
from .datashape import dshape
from .array import Array
from .constructors import array, empty, ones, zeros
from .eval import eval, append
from .persistence import open, drop, Storage
import ctypes

# These are so that llvm_structs corresponding to dshapes get converted correctly
#  when constructing ctypes functions
# FIXME:  They should probably go into a different module.
#   See blaze_kernels.py in _get_ctypes() method
class complex128(ctypes.Structure):
    _fields_ = [('real', ctypes.c_double),
                ('imag', ctypes.c_double)]
    _blaze_type_ = datashape.complex128

class complex64(ctypes.Structure):
    _fields_ = [('real', ctypes.c_float),
                ('imag', ctypes.c_float)]
    _blaze_type_ = datashape.complex64

__version__ = '0.0.1'

def print_versions():
    """Print all the versions of software that Blaze relies on."""
    import sys, platform
    import numpy as np
    from . import blz
    print("-=" * 38)
    print("Blaze version: %s" % __version__)
    print("NumPy version: %s" % np.__version__)
    try:
        import dynd
        print("DyND version: %s / LibDyND %s" %
                        (dynd.__version__, dynd.__libdynd_version__))
    except ImportError:
        print("DyND is not installed")
    print("BLZ version: %s" % blz.__version__)
    print("Blosc version: %s (%s)" % blz.blosc_version())
    print("Python version: %s" % sys.version)
    (sysname, nodename, release, version, machine, processor) = \
        platform.uname()
    print("Platform: %s-%s-%s (%s)" % (sysname, release, machine, version))
    if sysname == "Linux":
        print("Linux dist: %s" % " ".join(platform.linux_distribution()[:-1]))
    if not processor:
        processor = "not recognized"
    print("Processor: %s" % processor)
    print("Byte-ordering: %s" % sys.byteorder)
    print("Detected cores: %s" % blz.detect_number_of_cores())
    print("-=" * 38)


def test(verbosity=1, xunitfile=None, exit=False):
    """
    Runs the full Blaze test suite, outputting
    the results of the tests to sys.stdout.

    This uses nose tests to discover which tests to
    run, and runs tests in any 'tests' subdirectory
    within the Blaze module.

    Parameters
    ----------
        Value 0 prints very little, 1 prints a little bit,
        and 2 prints the test names while testing.
    xunitfile : string, optional
        If provided, writes the test results to an xunit
        style xml file. This is useful for running the tests
        in a CI server such as Jenkins.
    exit : bool, optional
        If True, the function will call sys.exit with an
        error code after the tests are finished.
    """
    import nose
    import os
    import sys
    argv = ['nosetests', '--verbosity=%d' % verbosity]
    # Output an xunit file if requested
    if xunitfile:
        argv.extend(['--with-xunit', '--xunit-file=%s' % xunitfile])
    # Add all 'tests' subdirectories to the options
    rootdir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(rootdir):
        if 'tests' in dirs:
            testsdir = os.path.join(root, 'tests')
            argv.append(testsdir)
            print('Test dir: %s' % testsdir[len(rootdir)+1:])
    # print versions (handy when reporting problems)
    print_versions()
    sys.stdout.flush()
    # Ask nose to do its thing
    return nose.main(argv=argv, exit=exit)
