from operator import eq

from bytei import ByteProvider
from idx import Indexable, AutoIndex, Space, Subspace, Index
from datashape.unification import union, can_embed

class Array(Indexable):
    """
    A numpy array without math functions
    """
    pass

class NDArray(Indexable):
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

class NDTable(Indexable):
    """
    A reified Table.
    """
    def __init__(self, obj, datashape=None, index=None, metadata=None):
        self.datashape = datashape
        self.metadata  = metadata
        self.space     = self.cast_space(obj)

        # NDTable always has an index
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
        Create a NDTable from a 1D list of byte providers.
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
        #import pdb; pdb.set_trace()

        # For example, the following sources would be regular

        #   A B C         A B C
        # 1 - - -   +  1  - - -
        #              2  - - -

        # TODO: there are other ways this could be true as well,
        # need more sophisticated checker
        regular = reduce(eq, shapes)

        # Indicate whether or not the union of the subspaces covers the
        # inner dimension.
        uni = reduce(union, shapes)

        # Does it cover the space?
        covers = map(can_embed, uni, shape)

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

        # Build the index for the space
        index = AutoIndex(shape, space)

        # this is perhaps IO side-effectful
        index.build()

        return NDTable(space, datashape=shape, index=index)

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
        return '<table><th>NDTable!</th></table>'

    def _repr_html_(self):
        return ('<div style="max-height:1000px;'
                'max-width:1500px;overflow:auto;">\n' +
                self.to_html() + '\n</div>')

    def __repr__(self):
        pass
