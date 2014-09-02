from __future__ import absolute_import, division, print_function

from datetime import date, datetime

from blaze.compute.core import *
from blaze import TableSymbol, by
from blaze.compatibility import raises

def test_errors():
    t = TableSymbol('t', '{foo: int}')
    with raises(NotImplementedError):
        compute_one(by(t, t.count()), 1)
