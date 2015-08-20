from odo import chunks
from blaze import discover, into, compute, symbol
from datashape.predicates import iscollection


L = [1, 2, 3, 4, 5, 6]
cL = chunks(list)([[1., 2., 3.], [4., 5., 6.]])
s = symbol('s', discover(cL))


def test_chunks_compute():
    exprs = [s, s + 1, s.max(), s.mean() + 1, s.head()]
    for e in exprs:
        result = compute(e, {s: cL})
        expected = compute(e, {s: L})
        if iscollection(e.dshape):
            result = into(list, result)
            expected = into(list, expected)
        assert result == expected


def test_chunks_head():
    assert compute(s.head(2), cL) == (1., 2.)


def test_pmap_default():
    flag = [0]

    def mymap(func, seq):
        flag[0] = True
        return map(func, seq)

    from blaze import set_default_pmap
    set_default_pmap(mymap)

    compute(s + 1, cL)

    assert flag[0] is True
