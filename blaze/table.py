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

from blaze.sources.descriptors.byteprovider import ByteProvider
from idx import Space, Subspace, Index

from datashape import DataShape, Fixed, dynamic, dshape as _dshape

from layouts.scalar import ChunkedL
from layouts.query import retrieve, write

from expr.graph import ArrayNode, injest_iterable
import metadata as md

from sources.chunked import CArraySource, CTableSource
from sources.canonical import ArraySource

from printer import generic_str, generic_repr

#------------------------------------------------------------------------
# Evaluation Class ( eclass )
#------------------------------------------------------------------------

MANIFEST = 1
DELAYED  = 2

#------------------------------------------------------------------------
# Indexable
#------------------------------------------------------------------------

# TODO: Indexable seems to be historical design notes, none of it
# is used in live code

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

    Parameters
    ----------

        obj : A list of byte providers, other NDTables or a Python object.

    Optional
    --------

        datashape : dshape
            Manual datashape specification for the table, if None then
            shape will be inferred if possible.
        metadata :
            Manual datashape specification for the table, if None then
            shape will be inferred if possible.

    Usage
    -----

        >>> Array([1,2,3])
        >>> Array([1,2,3], dshape='3, int32')
        >>> Array([1,2,3], dshape('3, int32'))
        >>> Array([1,2,3], params=params(clevel=3, storage='file'))

    """

    eclass = MANIFEST
    _metaheader = [
        md.manifest,
        md.arraylike,
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None,
            params=None):

        # Datashape
        # ---------

        if isinstance(dshape, basestring):
            dshape = _dshape(dshape)

        if not dshape:
            # The user just passed in a raw data source, try
            # and infer how it should be layed out or fail
            # back on dynamic types.
            self._datashape = dshape = CArraySource.infer_datashape(obj)
        else:
            # The user overlayed their custom dshape on this
            # data, check if it makes sense
            CArraySource.check_datashape(obj, given_dshape=dshape)
            self._datashape = dshape

        # Values
        # ------
        # Mimic NumPy behavior in that we have a variety of
        # possible arguments to the first argument which result
        # in different behavior for the values.

        if isinstance(obj, ByteProvider):
            self.data = obj
        else:
            self.data = CArraySource(obj, params=params)

        # children graph nodes
        self.children = []

        self.space = Space(self.data)

        # Layout
        # ------

        if layout:
            self._layout = layout
        elif not layout:
            self._layout = self.data.default_layout()

        # Metadata
        # --------

        self._metadata  = NDArray._metaheader + (metadata or [])

        # Parameters
        # ----------
        self.params = params


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
    def size(self):
        """
        Size of the Array.
        """
        # TODO: need to generalize, not every Array will look
        # like Numpy
        return sum(i.val for i in self._datashape.parameters[:-1])

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
        cc = self._layout.change_coordinates
        write(cc, indexer, value)

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self):
        raise NotImplementedError

    def __str__(self):
        return generic_str(self, deferred=False)

    def __repr__(self):
        return generic_repr('Array', self, deferred=False)


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

    def __init__(self, obj, dshape=None, metadata=None, layout=None,
            params=None):

        # Datashape
        # ---------

        if isinstance(dshape, basestring):
            dshape = _dshape(dshape)

        if not dshape:
            # The user just passed in a raw data source, try
            # and infer how it should be layed out or fail
            # back on dynamic types.
            self._datashape = dshape = CArraySource.infer_datashape(obj)
        else:
            # The user overlayed their custom dshape on this
            # data, check if it makes sense
            CArraySource.check_datashape(obj, given_dshape=dshape)
            self._datashape = dshape

        # Values
        # ------
        # Mimic NumPy behavior in that we have a variety of
        # possible arguments to the first argument which result
        # in different behavior for the values.

        if isinstance(obj, CArraySource):
            self.data = obj
        else:
            self.data = CArraySource(obj, params)

        # children graph nodes
        self.children = []

        self.space = Space(self.data)

        # Layout
        # ------

        if layout:
            self._layout = layout
        elif not layout:
            self._layout = ChunkedL(self.data, cdimension=0)

        # Metadata
        # --------

        self._metadata  = NDArray._metaheader + (metadata or [])

        # Parameters
        # ----------
        self.params = params


    def __str__(self):
        return generic_str(self, deferred=True)

    def __repr__(self):
        return generic_repr('NDArray', self, deferred=True)

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

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
    def size(self):
        """
        Size of the NDArray.
        """
        # TODO: need to generalize, not every Array will look
        # like Numpy
        return sum(i.val for i in self._datashape.parameters[:-1])

    @property
    def backends(self):
        """
        The storage backends that make up the space behind the
        Array.
        """
        return iter(self.space)

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


class Table(Indexable):

    eclass = MANIFEST
    _metaheader = [
        md.manifest,
        md.tablelike,
    ]

    def __init__(self, obj, dshape=None, metadata=None, layout=None,
            params=None):

        # Datashape
        # ---------

        if isinstance(dshape, basestring):
            dshape = _dshape(dshape)

        if not dshape:
            # The user just passed in a raw data source, try
            # and infer how it should be layed out or fail
            # back on dynamic types.
            self._datashape = dshape = CTableSource.infer_datashape(obj)
        else:
            # The user overlayed their custom dshape on this
            # data, check if it makes sense
            CTableSource.check_datashape(obj, given_dshape=dshape)
            self._datashape = dshape

        # Source
        # ------

        if isinstance(obj, ByteProvider):
            self.data = obj
        else:
            self.data = CTableSource(obj, dshape=dshape, params=params)

        # children graph nodes
        self.children = []

        self.space = Space(self.data)

        # Layout
        # ------

        if layout:
            self._layout = layout
        elif not layout:
            self._layout = self.data.default_layout()

        # Metadata
        # --------

        self._metadata  = NDTable._metaheader + (metadata or [])

        # Parameters
        # ----------
        self.params = params


class NDTable(Indexable, ArrayNode):
    """
    The base NDTable. Indexable contains the indexing logic for
    how to access elements, while ArrayNode contains the graph
    related logic for building expression trees with this table
    as an element.
    """

    eclass = DELAYED
    _metaheader = [
        md.deferred,
        md.tablelike,
    ]

    #------------------------------------------------------------------------
    # Properties
    #------------------------------------------------------------------------

    def __init__(self, obj, dshape=None, metadata=None, layout=None,
            params=None):

        # Datashape
        # ---------

        if isinstance(dshape, basestring):
            dshape = _dshape(dshape)

        if not dshape:
            # The user just passed in a raw data source, try
            # and infer how it should be layed out or fail
            # back on dynamic types.
            self._datashape = dshape = CTableSource.infer_datashape(obj)
        else:
            # The user overlayed their custom dshape on this
            # data, check if it makes sense
            CTableSource.check_datashape(obj, given_dshape=dshape)
            self._datashape = dshape

        # Source
        # ------

        if isinstance(obj, ByteProvider):
            self.data = obj
        else:
            self.data = CTableSource(obj, dshape=dshape, params=params)

        # children graph nodes
        self.children = []

        self.space = Space(self.data)

        # Layout
        # ------

        if layout:
            self._layout = layout
        elif not layout:
            self._layout = self.data.default_layout()

        # Metadata
        # --------

        self._metadata  = NDTable._metaheader + (metadata or [])

        # Parameters
        # ----------
        self.params = params

    @property
    def datashape(self):
        """
        Type deconstructor
        """
        return self._datashape

    @property
    def size(self):
        """
        Size of the NDTable.
        """
        # TODO: need to generalize, not every Array will look
        # like Numpy
        return sum(i.val for i in self._datashape.parameters[:-1])

    @property
    def backends(self):
        """
        The storage backends that make up the space behind the Array.
        """
        return iter(self.space)

    def __repr__(self):
        return generic_repr('NDTable', self, deferred=True)
