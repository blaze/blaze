from blaze import into, compute_up
from multipledispatch.conflict import ambiguities


def test_into_non_ambiguous():
    assert not ambiguities(into.funcs)


def test_compute_up_non_ambiguous():
    assert not ambiguities(compute_up.funcs)
