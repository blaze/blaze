"""
Usage:

# Region 1
r1 = [1,2,3,4]
# Region 2
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
x | 1 2 3 4   < -- Region a
y | 5 6 7 8   < -- Region b
"""

from idx import Index, AutoIndex, Region
from ndtable.adaptors.canonical import PyAdaptor, RawAdaptor

class NDTable(object):
    def __init__(self, obj, datashape, index=None, metadata=None):
        self.datashape = datashape
        self.metadata = metadata

        if isinstance(obj, list):
            self.index = AutoIndex(*obj)
        elif isinstance(obj, Index):
            self.index = index

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass

    @staticmethod
    def from_views(shape, *memviews):
        regions  = []
        indexes  = []

        adaptors = [RawAdaptor(mvw) for mvw in memviews]
        ntype    = shape[-1]

        for i, adapt in enumerate(adaptors):
            region   = Region(adapt)
            idx_size = region.size(ntype)

            regions += [region]
            indexes += [AutoIndex(idx_size)]

        # ???
        metadata = {}
        return NDTable(regions, shape, indexes, metadata)

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
