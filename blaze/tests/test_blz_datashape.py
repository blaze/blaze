""" Test blz datashapes

Tests datashapes supported in blz format
"""

from blaze import dshape, from_numpy



_bit_measures = [
    'bool', 
    'int', 'int8', 'int16', 'int32', 'int64',
    'uint', 'uint8', 'uint16', 'uint32', 'uint64',
    'float', 'float16', 'float32', 'float64',
    'complex', 'complex32', 'complex64', 'complex128'
]

_bit_like_measures = [
    'double',
    'string',
    'blob'
]

_shapes = [
    '10, ',
    '3, 5, ',
    '2, 3, 5, '
]

def check_datashape(ds):
    assert dshape(ds) == dshape(ds)

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
