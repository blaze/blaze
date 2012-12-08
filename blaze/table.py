"""
The toplevel modules containing the core Blaze datastructures.

    * Indexable
    * NDArray
    * NDTable
    * Table
    * Array

"""
import numpy as np
from operator import eq

from byteprovider import ByteProvider
from idx import Space, Subspace, Index

from datashape import DataShape, Fixed, dynamic, dshape as _dshape

from layouts.scalar import ChunkedL
from layouts.query import retrieve

from expr.graph import ArrayNode, injest_iterable
from expr import metadata as md

from sources.canonical import CArraySource, ArraySource
from printer import generic_str, generic_repr

#------------------------------------------------------------------------
# Evaluation Class ( eclass )
#------------------------------------------------------------------------

MANIFEST = 1
DELAYED  = 2

#------------------------------------------------------------------------
# Indexable
#------------------------------------------------------------------------

class Indexable(object):
    """
    The top abstraction in the Blaze class hierarchy.

    An index is a mapping from a domain specification to a collection of
    byte or subtables.  Indexable objects can be sliced/getitemed to
    return some other object in the Blaze system.
    """

    #------------------------------------------------------------------------
    # Slice/stride/getitem interface
    #
    # Define explicit indexing operations, as distinguished from operator
    # overloads, so that we can more easily guard, test, and validate
    # indexing calls based on their semantic intent, and not merely based
    # on the type of the operand.  Such dispatch may be done in the overloaded
    # operators, but that is a matter of syntactic sugar for end-user benefit.
    #------------------------------------------------------------------------

    def slice(self, slice_obj):
        """ Extracts a subset of values from this object. If there is
        no inner dimension, then this should return a scalar. Slicing
        typically preserves the data parallelism of the slicee, and the
        index-space transform is computable in constant time.
        """
        raise NotImplementedError

    def query(self, query_expr):
        """ Queries this object and produces a view or an actual copy
        of data (or a deferred eval object which can produce those). A
        query is typically a value-dependent streaming operation and
        produces an indeterminate number of return values.
        """
        raise NotImplementedError

    def take(self, indices, unique=None):
        """ Returns a view or copy of the indicated data.  **Indices**
        can be another Indexable or a Python iterable.  If **unique**
        if True, then implies that no indices are duplicated; if False,
        then implies that there are definitely duplicates.  If None, then
        no assumptions can be made about the indices.

        take() differs from slice() in that indices may be duplicated.
        """
        raise NotImplementedError

    #------------------------------------------------------------------------
    # Iteration protocol interface
    #
    # Defines the mechanisms by which other objects can determine the types
    # of iteration supported by this object.
    #------------------------------------------------------------------------

    def returntype(self):
        """ Returns the most efficient/general Data Descriptor this object can
        return.  Returns a value from the list the values defined in
        DataDescriptor.desctype: "buflist", "buffer", "streamlist", or
        "stream".
        """
        raise NotImplementedError
    def __index__(self):
        raise NotImplementedError()

    def global_id(self):
        "Get a unique global id for this source"
        # TODO: make it global :)
        return id(self)

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
        md.manifest,
        md.arraylike,
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None):

        self._datashape = dshape
        self._metadata  = Array._metaheader + (metadata or [])

        if isinstance(dshape, str):
            # run it through the parser
            dshape = _dshape(dshape)

        # Don't explictly check the dshape since we want to allow the
        # user to pass in anything that conforms to the dshape protocol
        # instead of requiring them to subclass.

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
        return generic_str(self, deferred=False)

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
        md.manifest,
        md.arraylike,
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None):

        self._datashape = dshape
        self._metadata  = NDArray._metaheader + (metadata or [])

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
            # Graph Nodes as input
            # ====================

            # XXX
            #self.children = injest_iterable(obj, force_homog=True)
            #self.space = Space(self.children)

            if not dshape:
                # The user just passed in a raw data source, try
                # and infer how it should be layed out or fail
                # back on dynamic types.
                self._datashape = CArraySource.infer_datashape(obj)
            else:
                # The user overlayed their custom dshape on this
                # data, check if it makes sense
                CArraySource.check_datashape(obj, given_dshape=dshape)
                self._datashape = dshape

            self.children = []
            self.vtype = self._datashape

            # Raw Data as input
            # ====================

            self.data = data = CArraySource(obj)
            self.space = Space(data)

            if not layout:
                # CArrays are always chunked on the first
                # dimension
                self._layout = ChunkedL(data, cdimension=0)

    def __str__(self):
        return generic_str(self, deferred=True)

    def __repr__(self):
        return generic_repr('NDArray', self, deferred=True)

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def mteretadata(self):
        return self._meta

    @property
    def name(self):
        return repr(self)

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
        md.deferred,
        md.tablelike,
    ]

    def __init__(self, obj, dshape=None, index=None, metadata=None):
        self._datashape = dshape
        self._metadata  = NDTable._metaheader + (metadata or [])

        if isinstance(dshape, str):
            # run it through the parser
            dshape = _dshape(dshape)

        # Resolve the values
        # ------------------
        if isinstance(obj, Space):
            self.space = obj
            self.children = set(self.space.subspaces)
        else:
            spaces = injest_iterable(obj)
            self.space = Space(*spaces)
            self.children = set(self.space.subspaces)

        # Resolve the shape
        # -----------------
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

        self._layout = None

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    @property
    def datashape(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def backends(self):
        """
        The storage backends that make up the space behind the Array.
        """
        return iter(self.space)

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
# Argument Munging
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
