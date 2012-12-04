from blaze.carray import carray, cparams
from bloscpack import pack_list, unpack_file
from numpy import array, frombuffer

def test_simple():
    filename = 'output'

    # hackish, just experimenting!
    arr = carray(xrange(10000)).chunks
    ca = [bytes(chunk.viewof) for chunk in arr]
    pack_list(ca, {}, filename, {'typesize': 8, 'clevel': 0, 'shuffle': False})

    out_list, meta_info = unpack_file('output')

    assert out_list[0] == ca[0]
    assert out_list[1] == ca[1]

def test_compressed():
    filename = 'output'

    # hackish, just experimenting!
    arr = carray(xrange(10000), cparams(clevel=5, shuffle=True)).chunks
    ca = [bytes(chunk.viewof) for chunk in arr]
    pack_list(ca, {}, filename, {'typesize': 8, 'clevel': 5, 'shuffle': True})

    out_list, meta_info = unpack_file('output')

    assert out_list[0] == ca[0]
    assert out_list[1] == ca[1]
