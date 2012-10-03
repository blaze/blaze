from datashape.coretypes import Fixed
from idx import Index, AutoIndex, Space, Subspace, can_coerce
from ndtable.adaptors.canonical import PyAdaptor, RawAdaptor
"""
Usage:

# Subspace 1
r1 = [1,2,3,4]
# Subspace 2
r2 = [5,6,7,8]

# Together these just form an amorphous lump of data without any
# structure.
a = PyAdaptor(r1)
b = PyAdaptor(r2)

rows    = ['x','y]
columns = ['a','b','c','d']

# The combination of these two properties is sufficient to
# overlay the desired structure on top of the lump of data.

idx = Index(rows, columns)
ds  = (2, 4, int32)

nd = NDTable([a,b], idx, ds)

    a b c d
  +---------
x | 1 2 3 4   < -- Subspace a
y | 5 6 7 8   < -- Subspace b
"""


class NDTable(object):
    def __init__(self, obj, datashape, index=None, metadata=None):
        self.datashape = datashape
        self.metadata = metadata

        if isinstance(obj, Space):
            self.space = obj
        else:
            assert False

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass

    @staticmethod
    def from_views(shape, *memviews):
        subspaces = []
        indexes   = []

        adaptors = [RawAdaptor(mvw) for mvw in memviews]
        ntype    = shape[-1]
        dimspec  = shape[0:-1]

        for i, adapt in enumerate(adaptors):
            subspace = Subspace(adapt)
            idx_size = subspace.size(ntype)

            axi = dimspec[-i]

            # If we have a fixed dimension specifier we need only
            # check that the element size of the array maps as a
            # multiple of the fixed value.
            if isinstance(axi, Fixed):
                assert idx_size == axi.val, (idx_size, axi.val)
                index_cls = AutoIndex

            # Otherwise we need more complex logic for example
            # Var(0, 5) would need to check that the bounds of
            # the subspace actually map onto range 0,5.
            else:
                index_cls = can_coerce(axi, idx_size, adapt)

            index_inst = index_cls(idx_size)

            subspaces += [subspace]
            indexes   += [index_inst]

        # ???
        metadata = {}

        space = Space(*subspaces)
        return NDTable(space, shape, indexes, metadata)

    @staticmethod
    def from_sql(dburl, query):
        pass

    @staticmethod
    def from_csv(fname, *params):
        pass

    # IPython notebook integration
    def to_html(self):
        return '<table><th>NDTable!</th></table>'

    def _repr_html_(self):
        return ('<div style="max-height:1000px;'
                'max-width:1500px;overflow:auto;">\n' +
                self.to_html() + '\n</div>')
