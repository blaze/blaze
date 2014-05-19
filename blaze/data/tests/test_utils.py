from __future__ import absolute_import, division, print_function

from blaze.data.utils import coerce
from collections import Iterator

def test_coerce():
    assert coerce('var * int', ['1', '2', '3']) == [1, 2, 3]

def test_coerce_lazy():
    assert list(coerce('var * int', iter(['1', '2', '3']))) == [1, 2, 3]
    assert isinstance(coerce('var * int', iter(['1', '2', '3'])), Iterator)
