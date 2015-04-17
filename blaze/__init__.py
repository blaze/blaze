from __future__ import absolute_import, division, print_function

try:
    import h5py  # if we import h5py after tables we segfault
except ImportError:
    pass

from pandas import DataFrame
from odo import odo, convert, append, resource, drop
from odo.backends.csv import CSV
from odo.backends.json import JSON, JSONLines

from multipledispatch import halt_ordering, restart_ordering

halt_ordering() # Turn off multipledispatch ordering

from datashape import dshape, discover
from .utils import ignoring
from .expr import (Symbol, TableSymbol, symbol, ndim, shape)
from .expr import (by, count, count_values, distinct, head, join, label, like,
        mean, merge, nunique, relabel, selection, sort, summary, var, transform)
from .expr import (date, datetime, day, hour, microsecond, millisecond, month,
        second, time, year)
from .expr.arrays import (tensordot, transpose)
from .expr.functions import *
from .index import create_index
from .interactive import *
from .compute.pmap import set_default_pmap
from .compute.csv import *
from .compute.json import *
from .compute.python import *
from .compute.pandas import *
from .compute.numpy import *
from .compute.core import *
from .compute.core import compute
from .cached import CachedDataset

with ignoring(ImportError):
    from .server import *
with ignoring(ImportError):
    from .sql import *
    from .compute.sql import *

with ignoring(ImportError, AttributeError):
    from .compute.spark import *
with ignoring(ImportError, TypeError):
    from .compute.sparksql import *
with ignoring(ImportError):
    from dynd import nd
    from .compute.dynd import *
with ignoring(ImportError):
    from .compute.h5py import *
with ignoring(ImportError):
    from .compute.hdfstore import *
with ignoring(ImportError):
    from .compute.pytables import *
with ignoring(ImportError):
    from .compute.chunks import *
with ignoring(ImportError):
    from .compute.bcolz import *
with ignoring(ImportError):
    from .mongo import *
    from .compute.mongo import *
with ignoring(ImportError):
    from .pytables import *
    from .compute.pytables import *

restart_ordering() # Restart multipledispatch ordering and do ordering

inf = float('inf')
nan = float('nan')

__version__ = '0.7.3'

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
