from blaze import into, drop, create_index
from blaze.compute.core import compute_down, compute_up
from multipledispatch.conflict import ambiguities
from blaze.expr.arithmetic import scalar_coerce


def test_no_dispatch_ambiguities():
    for func in [into, compute_up, compute_down, drop, create_index,
                 scalar_coerce]:
        assert not ambiguities(func.funcs)
