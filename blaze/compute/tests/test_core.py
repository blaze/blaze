from blaze.compute.core import *

def test_base():
    assert compute(1, 'anything') == 1
    assert compute(1.0, 'anything') == 1.0
    assert compute('hello', 'anything') == 'hello'
    assert compute(True, 'anything') == True
