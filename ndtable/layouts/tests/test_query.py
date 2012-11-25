from ndtable.layouts.scalar import IdentityL
from ndtable.sources.canonical import CArraySource
from ndtable.layouts.query import getitem

def test_simple():
    a = CArraySource([1,2,3])
    layout = IdentityL(a)
    indexer = (0,)

    getitem(layout.change_coordinates, indexer, a)
