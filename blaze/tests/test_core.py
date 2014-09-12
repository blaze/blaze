from blaze import into, compute_one
from multipledispatch.conflict import ambiguities


def test_into_non_ambiguous():
    assert not ambiguities(into.funcs)


def test_compute_one_non_ambiguous():
    assert not ambiguities(compute_one.funcs)
