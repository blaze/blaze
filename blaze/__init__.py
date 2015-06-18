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

halt_ordering()  # Turn off multipledispatch ordering

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
with ignoring(ImportError):
    from .compute.dask import *
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

restart_ordering()  # Restart multipledispatch ordering and do ordering

inf = float('inf')
nan = float('nan')

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
