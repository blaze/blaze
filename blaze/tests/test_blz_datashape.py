""" Test blz datashapes

Tests datashapes supported in blz format
"""
from blaze import dshape, from_numpy
from unittest import expectedFailure

# basic measures
# note that 'uint', 'complex' and 'complex32' do not exist
_bit_measures = [
    'bool', 
    'int', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float', 'float16', 'float32', 'float64',
    'complex64', 'complex128'
]

_bit_like_measures = [
    'double',
    'string',
    'blob'
]

_shapes = [
    '',
    '10, ',
    '3, 5, ',
    '2, 3, 5, '
]


_known_failures = [
    'double', '10, double', '3, 5, double', '2, 3, 5, double',
    'string', '10, string', '3, 5, string', '2, 3, 5, string'
]



def check_datashape_holds(ds):
    assert dshape(ds) == dshape(ds)


def check_datashape_fails(ds):
    try:
        assert not (dshape(ds) == dshape(ds))
    except:
        pass


def check_datashape(ds):
    if ds in _known_failures:
        check_datashape_fails(ds)
    else:
        check_datashape_holds(ds)

def test_blz_datashape_generator():
    for bm in _bit_measures:
        for shape in _shapes:
            yield check_datashape, shape + bm

    for blm in _bit_like_measures:
        for shape in _shapes:
            yield check_datashape, shape + blm


## Local Variables:
## mode: python
## coding: utf-8 
## python-indent: 4
## tab-width: 4
## fill-column: 66
## End:
