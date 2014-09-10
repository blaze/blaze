import numpy as np
import datashape as ds
import pytest

from blaze import into
from blaze.utils import tmpfile
from blaze.compatibility import xfail
from blaze.data.pytables import PyTables


tb = pytest.importorskip('tables')

now = np.datetime64('now').astype('datetime64[us]')


@pytest.fixture
def x():
    y = np.array([(1, 'Alice', 100),
                  (2, 'Bob', -200),
                  (3, 'Charlie', 300),
                  (4, 'Denis', 400),
                  (5, 'Edith', -500)],
                 dtype=[('id', '<i8'), ('name', 'S7'), ('amount', '<i8')])
    return y


@pytest.yield_fixture
def tbfile(x):
    with tmpfile('.h5') as filename:
        f = tb.open_file(filename, mode='w')
        d = f.create_table('/', 'title',  x)
        d.close()
        f.close()
        yield filename


@pytest.fixture
def raw_dt_data():
    raw_dt_data = [[1, 'Alice', 100, now],
                   [2, 'Bob', -200, now],
                   [3, 'Charlie', 300, now],
                   [4, 'Denis', 400, now],
                   [5, 'Edith', -500, now]]

    for i, d in enumerate(raw_dt_data):
        d[-1] += np.timedelta64(i, 'D')
    return list(map(tuple, raw_dt_data))


@pytest.fixture
def dt_data(raw_dt_data):
    return np.array(raw_dt_data, dtype=np.dtype([('id', 'i8'),
                                                 ('name', 'S7'),
                                                 ('amount', 'f8'),
                                                 ('date', 'M8[ms]')]))


@pytest.yield_fixture
def dt_tb(dt_data):
    class Desc(tb.IsDescription):
        id = tb.Int64Col(pos=0)
        name = tb.StringCol(itemsize=7, pos=1)
        amount = tb.Float64Col(pos=2)
        date = tb.Time64Col(pos=3)

    non_date_types = list(zip(['id', 'name', 'amount'], ['i8', 'S7', 'f8']))

    # has to be in microseconds as per pytables spec
    dtype = np.dtype(non_date_types + [('date', 'M8[us]')])
    rec = dt_data.astype(dtype)

    # also has to be a floating point number
    dtype = np.dtype(non_date_types + [('date', 'f8')])
    rec = rec.astype(dtype)
    rec['date'] /= 1e6
    with tmpfile('.h5') as filename:
        f = tb.open_file(filename, mode='w')
        d = f.create_table('/', 'dt', description=Desc)
        d.append(rec)
        d.close()
        f.close()
        yield filename


class TestPyTablesLight(object):
    def test_read(self, tbfile):
        assert PyTables(path=tbfile, datapath='/title').shape == (5,)

    def test_write_no_dshape(self, tbfile):
        with pytest.raises(ValueError):
            PyTables(path=tbfile, datapath='/write_this')

    @xfail(raises=NotImplementedError,
           reason='PyTables does not support object columns')
    def test_write_with_bad_dshape(self, tbfile):
        dshape = '{id: int, name: string, amount: float32}'
        PyTables(path=tbfile, datapath='/write_this', dshape=dshape)

    def test_write_with_dshape(self, tbfile):
        f = tb.open_file(tbfile, mode='a')
        try:
            assert '/write_this' not in f
        finally:
            f.close()
            del f

        # create our table
        dshape = '{id: int, name: string[7, "ascii"], amount: float32}'
        t = PyTables(path=tbfile, datapath='/write_this', dshape=dshape)

        assert t._v_file.filename == tbfile
        assert t.shape == (0,)

    def test_table_into_ndarray(self, dt_tb, dt_data):
        t = PyTables(dt_tb, '/dt')
        res = into(np.ndarray, t)
        for k in res.dtype.fields:
            lhs, rhs = res[k], dt_data[k]
            if (issubclass(np.datetime64, lhs.dtype.type) and
                issubclass(np.datetime64, rhs.dtype.type)):
                lhs, rhs = lhs.astype('M8[us]'), rhs.astype('M8[us]')
            assert np.array_equal(lhs, rhs)

    def test_ndarray_into_table(self, dt_tb, dt_data):
        dtype = ds.from_numpy(dt_data.shape, dt_data.dtype)
        t = PyTables(dt_tb, '/out', dtype)
        res = into(np.ndarray, into(t, dt_data, filename=dt_tb, datapath='/out'))
        for k in res.dtype.fields:
            lhs, rhs = res[k], dt_data[k]
            if (issubclass(np.datetime64, lhs.dtype.type) and
                issubclass(np.datetime64, rhs.dtype.type)):
                lhs, rhs = lhs.astype('M8[us]'), rhs.astype('M8[us]')
            assert np.array_equal(lhs, rhs)
