from operator import eq

from bytei import ByteProvider
from datashape.coretypes import Fixed, Var, TypeVar, DataShape
from idx import Indexable, AutoIndex, Space, Subspace, Index


class CannotEmbed(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "Cannot embed space of values (%r) in (%r)" % (
            self.space, self.dim
        )

def describe(obj):

    if isinstance(obj, DataShape):
        return obj

    elif isinstance(obj, list):
        return Fixed(len(obj))

    elif isinstance(obj, tuple):
        return Fixed(len(obj))

    elif isinstance(obj, DataTable):
        return obj.datashape

def can_embed(obj, dim2):
    """
    Can we embed a ``obj`` inside of the space specified by the outer
    dimension ``dim``.
    """
    dim1 = describe(obj)

    # We want explicit fallthrough
    if isinstance(dim1, Fixed):

        if isinstance(dim2, Fixed):
            if dim1 == dim2:
                return True

        if isinstance(dim2, Var):
            if dim2.lower < dim1.val < dim2.upper:
                return True

        if isinstance(dim2, TypeVar):
            return True

    if isinstance(dim1, TypeVar):
        return True

    raise CannotEmbed(dim1, dim2)

class IndexArray(Indexable):
    """
    A numpy array without math functions
    """
    pass

class Table(Indexable):
    """
    Deferred evaluation table that constructs the expression
    graph.
    """
    pass

class DataTable(Indexable):
    """
    A reified Table.
    """
    def __init__(self, obj, datashape=None, index=None, metadata=None):
        self.datashape = datashape
        self.metadata  = metadata
        self.space     = self.cast_space(obj)

        # DataTable always has an index
        if index is None:
            self.index = AutoIndex(self.space)
        elif isinstance(index, Index):
            self.index = index

    def cast_space(self, obj):

        if isinstance(obj, Space):
            space = obj
        elif isinstance(obj, Indexable):
            space = obj
        elif isinstance(obj, ByteProvider):
            space = obj
        elif isinstance(obj, list):
            space = map(self.add_space, obj)
        else:
            raise RuntimeError("Don't know how to cast")

        return space

    @staticmethod
    def from_providers(shape, *providers):
        """
        Create a DataTable from a 1D list of byte providers.
        """
        subspaces = []
        indexes   = []

        ntype    = shape[-1]
        outerdim = shape[0]
        innerdim = shape[1]

        # The number of providers must be compatable ( not neccessarily
        # equal ) with the number of given providers.
        assert can_embed(providers, outerdim)

        # Look at the metadata for the provider, see if we can
        # infer whether the given list of providers is regular
        shapes = [a.calculate(None) for a in providers]

        # For example, the following sources would be regular

        #   A B C         A B C
        # 1 - - -   +  1  - - -
        #              2  - - -

        # TODO: there are other ways this could be true as well,
        # need more sophisticated checker
        regular = reduce(eq, shapes)

        # Indicate whether or not the union of the subspaces covers the
        # inner dimension.
        covers = regular and (shapes[0] == innerdim)

        for i, provider in enumerate(providers):
            # Make sure we don't go over the outer dimension

            # (+1) because we don't usually consider 0 dimension
            # as 1
            assert (i+1) < outerdim

            subspace = Subspace(provider)
            substructure = subspace.size(ntype)

            # Can we embed the substructure inside of the of the inner
            # dimension?
            assert can_embed(substructure, innerdim)

            subspaces += [subspace]

        # ???
        metadata = {}

        space = Space(*subspaces)
        space.annotate(regular, covers)

        return DataTable(space, datashape=shape, index=None)

    def index1d(self, point):
        # Which subspace does the point exist in?
        preimage, x = self.index(point)

        # Return a 0 dimensional
        preimage.take()

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

    def __repr__(self):
        pass
