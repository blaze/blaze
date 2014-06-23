from blaze.compute.core import *
from datetime import date, datetime

def test_base():
    assert compute(1, 'anything') == 1
    assert compute(1.0, 'anything') == 1.0
    assert compute('hello', 'anything') == 'hello'
    assert compute(True, 'anything') == True
    assert compute(date(2012, 1, 1), 'anything') == date(2012, 1, 1)
