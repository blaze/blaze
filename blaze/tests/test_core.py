from blaze import into, compute_up, compute_down, drop, create_index
from multipledispatch.conflict import ambiguities


def test_no_dispatch_ambiguities():
    for func in [into, compute_up, compute_down, drop, create_index]:
        assert not ambiguities(func.funcs)
