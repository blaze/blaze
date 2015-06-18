import sys
import os
import json

import pytest

from datetime import datetime

import numpy as np
import pandas as pd
import h5py

from datashape import dshape

from blaze import discover
from blaze.utils import tmpfile, json_dumps


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


def test_json_encoder():
    result = json.dumps([1, datetime(2000, 1, 1, 12, 30, 0)],
                        default=json_dumps)
    assert result == '[1, "2000-01-01T12:30:00Z"]'
    assert json.loads(result) == [1, "2000-01-01T12:30:00Z"]
