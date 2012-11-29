from ndtable.layouts.scalar import ContinuousL
from ndtable.sources.canonical import CArraySource
from ndtable.layouts.query import retrieve

def test_simple():
    a = CArraySource([1,2,3])
    layout = ContinuousL(a)
    indexer = (0,)

    retrieve(layout.change_coordinates, indexer)
