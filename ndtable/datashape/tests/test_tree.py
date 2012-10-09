from ndtable.datashape.coretypes import expand, table_like, \
    array_like
from ndtable.datashape import parse

def test_expand():
    ex = parse('4, 3, 2, int32')

    top = list(expand(ex))
    assert len(top) == 4
    assert len(top[0]) == 3
    assert len(top[0][0]) == 2
    assert len(top[0][0][0]) == 1

    assert array_like(ex)

def test_expand_record():
    ex = parse('2, Record(x=int32, y=int32)')

    top = list(expand(ex))
    assert len(top) == 2
    assert top[0] == ['y','x']
    assert top[1] == ['y','x']

    assert table_like(ex)
