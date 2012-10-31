from operator import eq

from byteprovider import ByteProvider
from idx import Indexable, AutoIndex, Space, Subspace, Index

from datashape.unification import union
from datashape.coretypes import DataShape, Fixed

from ndtable.expr.graph import ArrayNode, injest_iterable

from carray import carray

def describe(obj):

    if isinstance(obj, DataShape):
        return obj

    if isinstance(obj, DataShape):
        return obj

    elif isinstance(obj, list):
        return Fixed(len(obj))

    elif isinstance(obj, tuple):
        return Fixed(len(obj))

    elif isinstance(obj, NDTable):
        return obj.datashape


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

#------------------------------------------------------------------------
# NDTable
#------------------------------------------------------------------------

# Here's how the multiple inheritance boils down.
#
#   Indexable
#   =========
#
#   index1d      : function
#   indexnd      : function
#   query        : function
#   returntype   : function
#   slice        : function
#   take         : function
#
#
#   ArrayNode
#   =========
#
#   children     : attribute
#   T            : function
#   dtype        : function
#   flags        : function
#   flat         : function
#   imag         : function
#   itemsize     : function
#   ndim         : function
#   real         : function
#   shape        : function
#   size         : function
#   strides      : function
#   tofile       : function
#   tolist       : function
#   tostring     : function
#   __len__      : function
#   __getitem__  : function
#   __getslice__ : function
#   __index__    : function

class NDTable(Indexable, ArrayNode):
    """
    The base NDTable. Indexable contains the indexing logic for
    how to access elements, while ArrayNode contains the graph
    related logic for building expression trees with this table
    as an element.

    Parameters:

        obj       : A list of byte providers, other NDTables or

    Optional:

        datashape : Manual datashape specification for the table,
                    if None then shape will be inferred if
                    possible.

        index     : The index for the datashape and all nested
                    structures, if None then AutoIndex is used.

        metadata  : Explicit metadata annotation.
    """

    def __init__(self, obj, datashape=None, index=None, metadata=None):
        self.datashape = datashape
        self.metadata  = metadata

        self.children = injest_iterable(obj)

        #self.space     = self.cast_space(obj)
        self.index = None

        # How do we build an Index from the given graph
        # elements... still needs some more thought. Disabled for
        # now.
        #
        # NDTable always has an index
        # if index is None:
        #     self.index = AutoIndex(self.space)
        # elif isinstance(index, Index):
        #     self.index = index

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
        Internal method to create a NDTable from a 1D list of
        byte providers. Tries to infer how the providers must be
        arranged in order to fit into the provided shape.
        """
        subspaces = []
        indexes   = []

        ntype    = shape[-1]
        outerdim = shape[0]
        innerdim = shape[1]


        provided_dim = describe(providers)

        # The number of providers must be compatable ( not neccessarily
        # equal ) with the number of given providers.

        # Look at the information for the provider, see if we can
        # infer whether the given list of providers is regular
        shapes = [p.calculate(ntype) for p in providers]

        # For example, the following sources would be regular

        #   A B C         A B C
        # 1 - - -   +  1  - - -
        #              2  - - -

        # TODO: there are other ways this could be true as well,
        # need more sophisticated checker
        regular = reduce(eq, shapes)
        covers = True

        # Indicate whether or not the union of the subspaces covers the
        # inner dimension.
        uni = reduce(union, shapes)

        # Does it cover the space?

        for i, provider in enumerate(providers):
            # Make sure we don't go over the outer dimension

            # (+1) because we don't usually consider 0 dimension
            # as 1
            assert (i+1) < outerdim

            subspace = Subspace(provider)
            substructure = subspace.size(ntype)

            # Can we embed the substructure inside of the of the inner
            # dimension?
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
        return 'NDTable %i' % id(self)

    @property
    def name(self):
        return repr(self)
