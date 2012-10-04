from sources.canonical import RawSource
from datashape.coretypes import Fixed
from idx import AutoIndex, Space, Subspace, can_coerce

class Indexable(object):
    """
    The top abstraction in the Blaze class hierarchy.
    """

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass

class IndexArray(object):
    """
    A numpy array without math functions
    """
    pass

class DataTable(object):
    def __init__(self, obj, datashape, index=None, metadata=None):
        self.datashape = datashape
        self.metadata = metadata

        if isinstance(obj, Space):
            self.space = obj
        else:
            assert False

    @staticmethod
    def from_views(shape, *memviews):
        """
        Create a DataTable from a 1D list of memoryviews.
        """
        subspaces = []
        indexes   = []

        adaptors = [RawSource(mvw) for mvw in memviews]

        # Just use the inner dimension
        ntype    = shape[-1]
        innerdim = shape[1]
        outerdim = shape[0]

        for i, adapt in enumerate(adaptors):
            # Make sure we don't go over the outer dimension
            assert i < outerdim

            subspace = Subspace(adapt)
            idx_size = subspace.size(ntype)

            # If we have a fixed dimension specifier we need only
            # check that the element size of the array maps as a
            # multiple of the fixed value.
            if isinstance(innerdim, Fixed):
                assert idx_size == innerdim.val, (idx_size, innerdim.val)
                index = AutoIndex(idx_size)

            # Otherwise we need more complex logic for example
            # Var(0, 5) would need to check that the bounds of
            # the subspace actually map onto the interval [0,5]
            else:
                index = can_coerce(innerdim, idx_size, adapt)

            subspaces += [subspace]
            indexes   += [index]

        # ???
        metadata = {}

        space = Space(*subspaces)
        return DataTable(space, shape, indexes, metadata)

    @staticmethod
    def from_sql(dburl, query):
        pass

    @staticmethod
    def from_csv(fname, *params):
        pass

    # IPython notebook integration
    def to_html(self):
        return '<table><th>DataTable!</th></table>'

    def _repr_html_(self):
        return ('<div style="max-height:1000px;'
                'max-width:1500px;overflow:auto;">\n' +
                self.to_html() + '\n</div>')
