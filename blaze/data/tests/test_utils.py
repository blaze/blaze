from __future__ import absolute_import, division, print_function

from blaze.data.utils import coerce, ordered_index
from collections import Iterator

def test_coerce():
    assert coerce('var * int', ['1', '2', '3']) == [1, 2, 3]

def test_coerce_lazy():
    assert list(coerce('var * int', iter(['1', '2', '3']))) == [1, 2, 3]
    assert isinstance(coerce('var * int', iter(['1', '2', '3'])), Iterator)

def test_ordered_index():
    index = (slice(None, None, None), 0)
    assert ordered_index(index, '3 * var * 2 * int32') == index
