from blaze.layouts.scalar import ContinuousL
from blaze.sources.canonical import CArraySource
from blaze.layouts.query import retrieve

def test_simple():
    a = CArraySource([1,2,3])
    layout = ContinuousL(a)
    indexer = (0,)

    retrieve(layout.change_coordinates, indexer)
