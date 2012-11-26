import numpy as np
from operator import eq

from byteprovider import ByteProvider
from idx import Indexable, Space, Subspace, Index

from datashape.coretypes import DataShape, Fixed
from layouts.scalar import ChunkedL
from layouts.query import retrieve
from slicealgebra import numpy_get

from expr.graph import ArrayNode, injest_iterable
from expr.metadata import metadata as md

from sources.canonical import CArraySource, ArraySource
from printer import generic_repr

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

        :obj: A list of byte providers, other NDTables or a Python object.

    Optional:

        :datashape: Manual datashape specification for the table,
                    if None then shape will be inferred if
                    possible.

        :metadata: Explicit metadata annotation.

    Usage:

        >>> Array([1,2,3])
        >>> Array([1,2,3], dshape='3, int32')
        >>> Array([1,2,3], dshape('3, 3, int32'))

    """

    eclass = MANIFEST
    _metaheader = [
        ('MANIFEST' , True),
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None):

        self._datashape = dshape
        self._metadata  = md.empty(
            self._metaheader + (metadata or [])
        )

        self.space = None

        # TODO: more robust
        if isinstance(obj, Indexable):
            infer_eclass(self._meta)

        elif isinstance(obj, str):
            # Create an empty array allocated per the datashape string
            self.space = None

        elif isinstance(obj, Space):
            self.space = obj

        else:
            if not dshape:
                # The user just passed in a raw data source, try
                # and infer how it should be layed out or fail
                # back on dynamic types.
                self._datashape = CArraySource.infer_datashape(obj)
            else:
                # The user overlayed their custom dshape on this
                # data, check if it makes sense
                if CArraySource.check_datashape(obj, given_dshape=dshape):
                    self._datashape = dshape
                else:
                    raise ValueError("Datashape is inconsistent with source")

            data = CArraySource(obj)
            self.space = Space(data)

            if not layout:
                # CArrays are always chunked on the first
                # dimension
                self._layout = ChunkedL(data, cdimension=0)

    @staticmethod
    def _from_providers(*providers):
        """
        Internal method to create a NDArray from a 1D list of byte
        providers. Tries to infer the simplest layout of how the
        providers fit together.
        """
        subspaces = [Subspace(provider) for provider in providers]
        space = Space(*subspaces)

        # TODO: more robust
        shape = providers[0].infer_datashape(providers[0])
        return Array(space, dshape=shape)

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

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
        cc = self._layout.change_coordinates
        return retrieve(cc, indexer)

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
        raise NotImplementedError

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
    _metaheader = [
        ('MANIFEST' , True),
        ('ARRAYLIKE', True),
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None):

        self._datashape = dshape
        self._metadata  = md.empty(
            self._metaheader + (metadata or [])
        )

        if isinstance(obj, str):
            # Create an empty array allocated per the datashape string
            self.space = None
            self.children = list(self.space.subspaces)

        elif isinstance(obj, Space):
            self.space = obj
            self.children = list(self.space.subspaces)

        if isinstance(obj, Indexable):
            infer_eclass(self._meta)

        else:
            # When no preference in backend specified, fall back on
            # Numpy and the trivial layout ( one buffer, dot product
            # stride formula )

            # Graph Nodes as input
            # ====================
            self.children = injest_iterable(obj, force_homog=True)
            self.space = Space(self.children)

            if not dshape:
                # The user just passed in a raw data source, try
                # and infer how it should be layed out or fail
                # back on dynamic types.
                self._datashape = CArraySource.infer_datashape(obj)
            else:
                # The user overlayed their custom dshape on this
                # data, check if it makes sense
                self._datashape = CArraySource.check_datashape(obj,
                    given_dshape=dshape)

            # Raw Data as input
            # ====================

            #data = CArraySource(obj)
            #self.space = Space(data)

            #if not layout:
                ## CArrays are always chunked on the first
                ## dimension
                #self._layout = ChunkedL(data, cdimension=0)

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return generic_repr('NDArray', self, deferred=True)


    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def metadata(self):
        return self._meta

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

    @staticmethod
    def _from_providers(*providers):
        """
        Internal method to create a NDArray from a 1D list of byte
        providers. Tries to infer the simplest layout of how the
        providers fit together.
        """
        subspaces = [Subspace(provider) for provider in providers]
        space = Space(*subspaces)

        # TODO: more robust
        shape = providers[0].infer_datashape(providers[0])
        return NDArray(space, dshape=shape)


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
    _metaheader = [
        ('DEFERRED' , True),
    ]

    def __init__(self, obj, datashape=None, index=None, metadata=None):
        self._datashape = datashape
        self._metadata  = md.empty(
            self._metaheader + (metadata or [])
        )

        if isinstance(obj, Space):
            self._space = obj
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
    def _from_providers(*providers):
        """
        Injest providers and cast them into columns.
        """
        subspaces = [Subspace(provider) for provider in providers]
        space = Space(*subspaces)

        shape = providers[0].infer_datashape(providers[0])
        return NDTable(space, datashape=shape)

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
