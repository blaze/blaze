import pytest

nb = pytest.importorskip('numba')

from numba import float64, int64, float32
from numba.types import NPDatetime as datetime64, NPTimedelta as timedelta64
from numba.types import CharSeq as char, UnicodeCharSeq as unichar
from blaze.compute.numba import compute_signature, get_numba_type
from blaze import symbol
import datashape


def test_compute_signature():
    s = symbol('s', 'int64')
    t = symbol('t', 'float32')
    d = symbol('d', 'datetime')

    assert compute_signature(s + t) == float64(int64, float32)
    assert (compute_signature(d.truncate(days=1)) ==
            datetime64('D')(datetime64('us')))
    assert compute_signature(d.day + 1) == int64(datetime64('us'))


def test_get_numba_type():
    assert get_numba_type(datashape.bool_) == nb.bool_
    assert get_numba_type(datashape.date_) == datetime64('D')
    assert get_numba_type(datashape.datetime_) == datetime64('us')
    assert get_numba_type(datashape.timedelta_) == timedelta64('us')
    assert get_numba_type(datashape.TimeDelta('D')) == timedelta64('D')
    assert get_numba_type(datashape.int64) == int64
    assert get_numba_type(datashape.String(7, "A")) == char(7)
    assert get_numba_type(datashape.String(None, "A")) == nb.types.string
    assert get_numba_type(datashape.String(7)) == unichar(7)


def test_fail_on_object_type():
    with pytest.raises(TypeError):
        get_numba_type(datashape.object_)


@pytest.mark.xfail(raises=TypeError,
                   reason="Cannot infer variable length string type yet")
def test_get_numba_type_failures():
    get_numba_type(datashape.string)


@pytest.mark.xfail(raises=TypeError,
                   reason='Cannot infer type of record dshapes yet')
def test_get_record_type():
    get_numba_type(datashape.dshape('10 * {a: int64}'))
