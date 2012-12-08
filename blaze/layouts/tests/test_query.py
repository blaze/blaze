"""
Test querying data from a data source with the overlayed layout
"""

from blaze.layouts.query import retrieve
from blaze.layouts.scalar import ContiguousL
from blaze.sources.chunked import CArraySource
from blaze.sources.canonical import ArraySource


def test_simple():
    a = CArraySource([1,2,3])
    layout = ContiguousL(a)
    indexer = (0,)

    retrieve(layout.change_coordinates, indexer)

#def test_numpy():
    #a = ArraySource([1,2,3])
    #layout = ContigiousL(a)
    #indexer = (0,)

    #retrieve(layout.change_coordinates, indexer)

#def test_numpy_strided():
    #a = ArraySource([1,2,3])
    #layout = StridedL((8,8))
    #indexer = (0,)

    #retrieve(layout.change_coordinates, indexer)
