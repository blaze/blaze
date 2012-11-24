import numpy as np
from operator import eq

from byteprovider import ByteProvider
from idx import Indexable, AutoIndex, Space, Subspace, Index

from datashape.coretypes import DataShape, Fixed, from_numpy
from regions.scalar import IdentityL
from slicealgebra import numpy_get

from expr.graph import ArrayNode, injest_iterable
from expr.metadata import metadata as md

from sources.canonical import CArraySource, ArraySource
from printer import array2string, table2string, generic_repr

#------------------------------------------------------------------------
# Evaluation Class ( eclass )
#------------------------------------------------------------------------

MANIFEST = 1
DELAYED  = 2

#------------------------------------------------------------------------
# Immediate
#------------------------------------------------------------------------

class Array(Indexable):
    """
    Manifest array, does not create a graph. Forces evaluation on every
    call.

    Parameters:

        obj       : A list of byte providers, other NDTables or
                    a Python object.

    Optional:

        datashape : Manual datashape specification for the table,
                    if None then shape will be inferred if
                    possible.

        metadata  : Explicit metadata annotation.

    Example:

        >>> Array([1,2,3])
        >>> Array([1,2,3], dshape='3, int32')

    """

    eclass = MANIFEST

    def __init__(self, obj, dshape=None, metadata=None):

        self._datashape = dshape
        self._metadata  = metadata or md.empty()
        self._layout    = None

        # The "value space"
        self.space      = None

        if isinstance(obj, Indexable):
            infer_eclass(self._meta)

        elif isinstance(obj, str):
            # Create an empty array allocated per the datashape string
            self.space = None

        elif isinstance(obj, Space):
            self.space = obj

        else:
            # When no preference in backend specified, fall back on
            # Numpy backend and the trivial layout ( one covering space,
            # dot product stride formula )
            assert isinstance(obj, list)

            ca = CArraySource(obj)
            self.space = Space(ca)
            self._datashape = ca.infer_datashape()
            self._layout = IdentityL(ca)

            # -- The Future --
            # The general case, we'll get there eventually...
            # for now just assume that we're passing in a
            # Python list that we wrap around Numpy.
            #self.children = injest_iterable(obj, force_homog=True)

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def _meta(self):
        """
        Intrinsic metadata associated with this class of object
        """
        return md({
            'ECLASS': self.eclass,
        })

    @property
    def metadata(self):
        return self._metadata + self._meta

    @property
    def type(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def datashape(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def backends(self):
        """
        The storage backends that make up the space behind the
        Array.
        """
        return iter(self.space)

    #------------------------------------------------------------------------
    # Basic Slicing
    #------------------------------------------------------------------------

    # Immediete slicing
    def __getitem__(self, indexer):
        # ---------------
        # First Transform
        # ---------------

        # Pass the indexer object through the coordinate
        # transformations of the layout backend.

        # This returns the ByteProvider block to be operated on and the
        # coordinates with respect to the block.

        # ===================================
        block, coords = self._layout[indexer]
        # ===================================

        # ----------------
        # Second Transform
        # ----------------

        # Infer the slicealgebra system we need to use to dig
        # into the block in question. I.e. for Numpy this is the
        # general dot product

        plan = block.ca[indexer]

        # ----------------
        # Third Transform
        # ----------------
        # Aggregate the data from the Data Descriptors and pass
        # them into a new Array object built around the result.
        # Infer the new coordinates of this resulting block.

        return block

    # Immediete slicing ( Side-effectful )
    def __setitem__(self, indexer, value):
        if isinstance(indexer, slice):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self):
        raise NotImplementedError

    def __str__(self):
        return array2string(self)

    def __repr__(self):
        return generic_repr('Array', self, deferred=False)

class Table(Indexable):
    pass

#------------------------------------------------------------------------
# Deferred
#------------------------------------------------------------------------

class NDArray(Indexable, ArrayNode):
    """
    Deferred array, operations on this array create a graph built
    around an ArrayNode.
    """

    eclass = DELAYED

    def __init__(self, obj, datashape=None, metadata=None):

        self._datashape = datashape
        self._metadata  = metadata or md.empty()

        if isinstance(obj, str):
            # Create an empty array allocated per the datashape string
            self.space = None
            self.children = list(self.space.subspaces)

        elif isinstance(obj, Space):
            self.space = obj
            self.children = list(self.space.subspaces)

        else:
            # When no preference in backend specified, fall back on
            # Numpy and the trivial layout ( one buffer, dot product
            # stride formula )
            self.children = injest_iterable(obj, force_homog=True)
            self.space = Space(self.children)

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def _meta(self):
        """
        Intrinsic metadata associated with this class of object
        """
        return md({
            'ECLASS': self.eclass,
        })

    @property
    def metadata(self):
        return self._metadata + self._meta

    @property
    def name(self):
        return repr(self)

    # TODO: deprecate this
    @property
    def type(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def datashape(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def backends(self):
        """
        The storage backends that make up the space behind the
        Array.
        """
        return iter(self.space)

    #------------------------------------------------------------------------
    # Alternative Constructors
    #------------------------------------------------------------------------

    @staticmethod
    def from_providers(shape, *providers):
        """
        Internal method to create a NDArray from a 1D list of
        byte providers. Tries to infer how the providers must be
        arranged in order to fit into the provided shape.
        """

        # TODO: now we just infer how we layout the providers
        # passed in
        subspaces = []

        ntype    = shape[-1]
        outerdim = shape[0]
        innerdim = shape[1]

        #provided_dim = cast_arguments(providers)

        # TODO: what now?
        #shapes = [p.calculate(ntype) for p in providers]

        #regular = reduce(eq, shapes)
        #covers = True

        #uni = reduce(union, shapes)

        for i, provider in enumerate(providers):
            # Make sure we don't go over the outer dimension

            # (+1) because we don't usually consider 0 dimension
            # as 1
            assert (i+1) < outerdim

            subspace = Subspace(provider)
            subspaces += [subspace]

        space = Space(*subspaces)
        #space.annotate(regular, covers)

        return NDArray(space, datashape=shape)

    def __str__(self):
        return array2string(self)

    def __repr__(self):
        return generic_repr('NDArray', self, deferred=True)


#------------------------------------------------------------------------
# NDTable
#------------------------------------------------------------------------

# Here's how the multiple inheritance boils down. Going to remove
# the multiple inheritance at some point because it's not kind
# for other developers.
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
#   __index__    : function

class NDTable(Indexable, ArrayNode):
    """
    The base NDTable. Indexable contains the indexing logic for
    how to access elements, while ArrayNode contains the graph
    related logic for building expression trees with this table
    as an element.

    Parameters:

        obj       : A list of byte providers, other NDTables or
                    a Python list.

    Optional:

        datashape : Manual datashape specification for the table,
                    if None then shape will be inferred if
                    possible.

        index     : The index for the datashape and all nested
                    structures, if None then AutoIndex is used.

        metadata  : Explicit metadata annotation.

    """

    eclass = DELAYED

    def __init__(self, obj, datashape=None, index=None, metadata=None):
        self._datashape = datashape
        self._metadata  = metadata or md.empty()

        if isinstance(obj, Space):
            self.space = obj
            self.children = set(self.space.subspaces)
        else:
            self.children = injest_iterable(obj)

        # How do we build an Index from the given graph
        # elements... still needs some more thought. Disabled for
        # now.
        #
        # NDTable always has an index

        #if index is None:
            #self.index = AutoIndex(self.space)
        #elif isinstance(index, Index):
            #self.index = index

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def _meta(self):
        """
        Intrinsic metadata associated with this class of object
        """
        return md({
            'ECLASS': self.eclass,
        })

    @property
    def metadata(self):
        return self._metadata + self._meta

    @property
    def type(self):
        """
        """
        return self._datashape

    @property
    def datashape(self):
        """
        Type deconstructor
        """
        return self._datashape

    #------------------------------------------------------------------------
    # Construction
    #------------------------------------------------------------------------

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

        provided_dim = cast_arguments(providers)

        # The number of providers must be compatable ( not neccessarily
        # equal ) with the number of given providers.

        # Look at the information for the provider, see if we can
        # infer whether the given list of providers is regular
        #shapes = [p.calculate(ntype) for p in providers]

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
        #uni = reduce(union, shapes)

        # Does it cover the space?

        for i, provider in enumerate(providers):
            # Make sure we don't go over the outer dimension

            # (+1) because we don't usually consider 0 dimension
            # as 1
            assert (i+1) < outerdim

            subspace = Subspace(provider)

            # Can we embed the substructure inside of the of the inner
            # dimension?
            subspaces += [subspace]

        space = Space(*subspaces)
        space.annotate(regular, covers)

        # Build the index for the space
        index = AutoIndex(shape, space)

        # this is perhaps IO side-effectful
        index.build()

        return NDTable(space, datashape=shape, index=index)

    def __repr__(self):
        return generic_repr('NDTable', self, deferred=True)

    #------------------------------------------------------------------------
    # Convenience Methods
    #------------------------------------------------------------------------

    @staticmethod
    def from_sql(dburl, query):
        pass

    @staticmethod
    def from_csv(fname, *params):
        pass

# EClass are closed under operations. Operations on Manifest
# arrays always return other manifest arrays, operations on
# delayed arrays always return delayed arrays. Mixed operations
# may or may not be well-defined.

#     EClass := { Manifest, Delayed }
#
#     a: EClass, b: EClass, f : x -> y, x: a  |-  f(x) : y
#     f : x -> y,  x: Manifest  |-  f(x) : Manifest
#     f : x -> y,  x: Delayed   |-  f(x) : Delayed

# TBD: Constructor have these closure properties?? Think about
# this more...

# C: (t0, t1) -> Manifest, A: Delayed, B: Delayed,  |- C(A,B) : Manifest
# C: (t0, t1) -> Delayed, A: Manifest, B: Manifest, |- C(A,B) : # Delayed

# This is a metadata space transformation that informs the
# codomain eclass judgement.
def infer_eclass(a,b):
    if (a,b) == (MANIFEST, MANIFEST):
        return MANIFEST
    if (a,b) == (MANIFEST, DELAYED):
        return MANIFEST
    if (a,b) == (DELAYED, MANIFEST):
        return MANIFEST
    if (a,b) == (DELAYED, DELAYED):
        return DELAYED

#------------------------------------------------------------------------
# Deconstructors
#------------------------------------------------------------------------

def cast_arguments(obj):
    """
    Handle different sets of arguments for constructors.
    """

    if isinstance(obj, DataShape):
        return obj

    elif isinstance(obj, list):
        return Fixed(len(obj))

    elif isinstance(obj, tuple):
        return Fixed(len(obj))

    elif isinstance(obj, NDTable):
        return obj.datashape
