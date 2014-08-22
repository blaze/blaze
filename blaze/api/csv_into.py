from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
from datashape import DataShape, dshape, Record, to_numpy_dtype
import toolz
from toolz import concat, partition_all, valmap
from cytoolz import pluck
import copy
from datetime import datetime
from datashape.user import validate, issubschema
from numbers import Number
from collections import Iterable, Iterator
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import h5py
import tables

from ..dispatch import dispatch
from ..expr.table import TableExpr
from ..compute.core import compute


__all__ = ['csv_into']


def csv_into(a, b, permit_errors=False, **kwargs):

    if (permit_errors):
        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = cap_stderr = StringIO()
        ans = _csv_into(a, b, **kwargs)
        sys.stderr = cap_stderr
        cap_stderr.seek(0)
        return ans, cap_stderr.read()
    else:
        return _csv_into(a, b, **kwargs)


@dispatch(object, object)
def _csv_into(a, b, **kwargs):
    """
    Push CSV-like data at pth into a container of type ``a``

    """
    raise NotImplementedError(
        "Blaze does not know a rule for the following conversion"
        "\n%s <- %s" % (type(a).__name__, type(b).__name__))


@dispatch(DataFrame, str)
def _csv_into(a, b, **kwargs):

    import os
    if os.path.isfile(b):
        return pd.read_csv(b, header=None, error_bad_lines=False,
                           warn_bad_lines=True)
    elif os.path.isdir(b):
        allFiles = glob.glob(os.path.join(b, "/*.csv"))
        frame = pd.DataFrame()
        alldfs = []
        for f in allFiles:
            df = pd.read_csv(f, header=None, error_bad_lines=False,
                             warn_bad_lines=True)
            alldfs.append(df)
        return pd.concat(alldfs)
