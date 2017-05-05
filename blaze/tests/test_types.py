import dask.array as da
import numpy as np
import pandas as pd
import pytest

from blaze.interactive import into
from blaze.types import (
    iscorescalar,
    iscoresequence,
    iscoretype,
)


@pytest.mark.parametrize('data,res',
                         [(1, True),
                          (1.1, True),
                          ("foo", True),
                          ([1, 2], False),
                          ((1, 2), False),
                          (pd.Series([1, 2]), False)])
def test_iscorescalar(data, res):
    assert iscorescalar(data) == res


@pytest.mark.parametrize('data,res',
                         [(1, False),
                          ("foo", False),
                          ([1, 2], True),
                          ((1, 2), True),
                          (pd.Series([1, 2]), True),
                          (pd.DataFrame([[1, 2], [3, 4]]), True),
                          (np.ndarray([1, 2]), True),
                          (into(da.core.Array, [1, 2], chunks=(10,)), False)])
def test_iscoresequence(data, res):
    assert iscoresequence(data) == res


@pytest.mark.parametrize('data,res',
                         [(1, True),
                          ("foo", True),
                          ([1, 2], True),
                          ((1, 2), True),
                          (pd.Series([1, 2]), True),
                          (pd.DataFrame([[1, 2], [3, 4]]), True),
                          (np.ndarray([1, 2]), True),
                          (into(da.core.Array, [1, 2], chunks=(10,)), False)])
def test_iscoretype(data, res):
    assert iscoretype(data) == res
