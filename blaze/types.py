import datetime
import decimal

import numpy as np
import pandas as pd

CORE_SCALAR_TYPES = (
    float,
    decimal.Decimal,
    int,
    bool,
    str,
    pd.Timestamp,
    datetime.date,
    datetime.timedelta
)
CORE_SEQUENCE_TYPES = (
    list,
    dict,
    tuple,
    set,
    pd.Series,
    pd.DataFrame,
    np.ndarray
)
CORE_TYPES = CORE_SCALAR_TYPES + CORE_SEQUENCE_TYPES


def iscorescalar(x):
    return isinstance(x, CORE_SCALAR_TYPES)


def iscoresequence(x):
    return isinstance(x, CORE_SEQUENCE_TYPES)


def iscoretype(x):
    return isinstance(x, CORE_TYPES)
