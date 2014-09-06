
import sys
import pytest
from functools import partial
from blaze.data import CSV, JSON

from blaze.utils import tmpfile, raises
from blaze.data.utils import tuplify
from blaze.compatibility import xfail, PY2, PY3

import gzip

is_py2_win = sys.platform == 'win32' and PY2


@pytest.yield_fixture
def rsrc():
    with tmpfile('.csv.gz') as filename:
        f = gzip.open(filename, 'wt')
        f.write('1,1\n2,2')
        f.close()
        yield filename


@xfail(reason='Invalid opener')
def test_gzopen_no_gzip_open(rsrc):
    dd = CSV(rsrc, schema='2 * int')
    assert tuplify(list(dd)) == ((1, 1), (2, 2))


@xfail(is_py2_win, reason='Win32 py2.7 unicode/gzip/eol needs sorting out')
def test_gzopen_csv(rsrc):
    dd = CSV(rsrc, schema='2 * int', open=partial(gzip.open, mode='rt'))
    assert tuplify(list(dd)) == ((1, 1), (2, 2))


@xfail(is_py2_win, reason='Win32 py2.7 unicode/gzip/eol needs sorting out')
def test_gzopen_json():
    with tmpfile('.json.gz') as filename:
        f = gzip.open(filename, 'wt')
        f.write('[[1, 1], [2, 2]]')
        f.close()

        # Not a valid JSON file
        assert raises(Exception, lambda: list(JSON(filename, schema='2 * int')))

        dd = JSON(filename, schema='2 * int', open=gzip.open)

        assert tuplify(list(dd)) == ((1, 1), (2, 2))
