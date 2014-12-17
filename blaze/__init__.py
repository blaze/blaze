from __future__ import absolute_import, division, print_function

from pandas import DataFrame
from into import into, convert, append, resource, drop
from into.backends.csv import CSV

from multipledispatch import halt_ordering, restart_ordering

halt_ordering() # Turn off multipledispatch ordering

from datashape import dshape, discover
from .expr import (Symbol, TableSymbol, symbol)
from .expr import (by, count, count_values, distinct, head, join, label, like,
        mean, merge, nunique, relabel, selection, sort, summary, var, transform)
from .expr import (date, datetime, day, hour, microsecond, millisecond, month,
        second, time, year)
from .expr.functions import *
from .index import create_index
from .json import *
from .interactive import *
from .compute.csv import *
from .compute.python import *
from .compute.pandas import *
from .compute.numpy import *
from .compute.core import *
from .compute.core import compute

try:
    from .server import *
except ImportError:
    pass

try:
    from .sql import *
    from .compute.sql import *
except ImportError:
    pass

try:
    from .spark import *
except (AttributeError, ImportError):
    pass
try:
    from .compute.sparksql import *
    from .sparksql import *
except (ImportError, TypeError):
    pass
try:
    from dynd import nd
    from .compute.dynd import *
except ImportError:
    pass
try:
    from .h5py import *
    from .compute.h5py import *
except ImportError:
    pass
try:
    from .compute.pytables import *
except ImportError:
    pass
try:
    from .compute.chunks import *
except ImportError:
    pass
try:
    from .compute.bcolz import *
except ImportError:
    pass
try:
    from .mongo import *
    from .compute.mongo import *
except ImportError:
    pass
try:
    from .pytables import *
    from .compute.pytables import *
except ImportError:
    pass

restart_ordering() # Restart multipledispatch ordering and do ordering

inf = float('inf')
nan = float('nan')

__version__ = '0.7.0'

# If IPython is already loaded, register the Blaze catalog magic
# from . import catalog
# import sys
# if 'IPython' in sys.modules:
#     catalog.register_ipy_magic()
# del sys

def print_versions():
    """Print all the versions of software that Blaze relies on."""
    import sys, platform
    import numpy as np
    import datashape
    print("-=" * 38)
    print("Blaze version: %s" % __version__)
    print("Datashape version: %s" % datashape.__version__)
    print("NumPy version: %s" % np.__version__)
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
    print("-=" * 38)


def test(verbose=False, junitfile=None, exit=False):
    """
    Runs the full Blaze test suite, outputting
    the results of the tests to sys.stdout.

    This uses py.test to discover which tests to
    run, and runs tests in any 'tests' subdirectory
    within the Blaze module.

    Parameters
    ----------
    verbose : int, optional
        Value 0 prints very little, 1 prints a little bit,
        and 2 prints the test names while testing.
    junitfile : string, optional
        If provided, writes the test results to an junit xml
        style xml file. This is useful for running the tests
        in a CI server such as Jenkins.
    exit : bool, optional
        If True, the function will call sys.exit with an
        error code after the tests are finished.
    """
    import os
    import sys
    import pytest

    args = []

    if verbose:
        args.append('--verbose')

    # Output an xunit file if requested
    if junitfile is not None:
        args.append('--junit-xml=%s' % junitfile)

    # Add all 'tests' subdirectories to the options
    rootdir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(rootdir):
        if 'tests' in dirs:
            testsdir = os.path.join(root, 'tests')
            args.append(testsdir)
            print('Test dir: %s' % testsdir[len(rootdir) + 1:])
    # print versions (handy when reporting problems)
    print_versions()
    sys.stdout.flush()

    # Ask pytest to do its thing
    error_code = pytest.main(args=args)
    if exit:
        return sys.exit(error_code)
    return error_code == 0
