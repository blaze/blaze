from ..expr.table import TableSymbol
from ..data.core import DataDescriptor
from ..data.python import Python
from ..dispatch import dispatch
from datashape import discover
import itertools

names = ('_%d' % i for i in itertools.count(1))

class Table(TableSymbol):
    """ Interactive Table """
    __slots__ = 'data', 'schema', 'name'

    @dispatch(DataDescriptor)
    def __init__(self, dd, name=None):
        self.data = dd
        self.schema = dd.schema
        self.name = name or next(names)

    @dispatch((list, tuple))
    def __init__(self, seq, name=None):
        self.schema = discover(seq[:10]).subshape[0]
        self.data = tuple(seq)
        self.name = name or next(names)

    def resources(self):
        return {self: self.data}


@dispatch(Table)
def compute(t):
    return t.data
