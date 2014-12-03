
from blaze import discover, into, drop, resource
import h5py
from contextlib import contextmanager
from blaze.h5py import *
from blaze.utils import tmpfile

data = [(1, 32.4, 'Alice'),
        (2, 234.24, 'Bob'),
        (4, -430.0, 'Joe')]


x = np.array(data, dtype=[('id', int), ('amount', float), ('name', str, 100)])

ds = dshape("3 * {id : int64, amount: float64, name: string}")


def eq(a, b):
    c = a == b
    if isinstance(c, np.ndarray):
        c = c.all()
    return c


@contextmanager
def h():
    with tmpfile('.hdf5') as f:
        f = h5py.File(f)
        fx = f.create_dataset('/x', shape=x.shape, dtype=x.dtype, chunks=True,
                maxshape=(None,))
        fx[:] = x

        yield f


def test_discover():
    with h() as fx:
        assert str(discover(fx)) == str(discover({'x': x}))


def test_hdf5_from_datashape():
    with tmpfile('.hdf5') as fn:
        f = hdf5_from_datashape(fn, '{x: int32, y: {z: 3 * int32}}')
        assert isinstance(f, h5py.File)
        assert 'x' in f
        assert f['y/z'].shape == (3,)
        assert f['y/z'].dtype == 'i4'

        # ensure idempotence
        f = hdf5_from_datashape(fn, '{x: int32, y: {z: 3 * int32}}')


def test_resource():
    with h() as f:
        assert resource(f.filename).keys() == f.keys()

def test_resource_creates_dshape_if_necessary():
    with tmpfile('.hdf5') as fn:
        f = resource(fn, dshape='{x: int32, y: {z: 3 * int32}}')
        assert f['y']['z'].shape == (3,)


def test_into_from_numpy_array():
    with h() as f:
        d = into(f['x'], x)
        assert eq(d[:], np.concatenate([x, x]))

def test_into_from_iterator():
    with h() as f:
        d = into(f['x'], data)
        assert eq(d[:], np.concatenate([x, x]))


def test_ndarrays():
    x = np.ones((5, 5), dtype='i4')
    with tmpfile('.hdf5') as fn:
        d = into(fn + '::/x', x)
        assert isinstance(d, h5py.Dataset)
        assert d.file.filename == fn
