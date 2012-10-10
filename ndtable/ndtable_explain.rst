
======================
Blaze Concept Overview
======================

                        Indexable
                            |
                    +-------+------+
                    |              |
      Shape --> DataSpace     ByteProvider --> DataDescriptor
                    |                                |
          +---------+--------+               +----+--+--+----+
          |                  |               |    |     |    | 
       Tabular           ArrayLike         Buffer |   Stream |
          |                  |                 BufList   StreamList
     +----+----+        +----+----+
     |         |        |         |
   Table    NDTable   Array    NDArray


Indexable:
    Any object which supports indexing by scalars to retrieve individual
    elements, and slicing (indexing by more complex objects, usually Tuples)
    to retrieve subsets or subspaces.  There is no concept of "shape" here in
    the classical Numpy sense; an Indexable should be considered to be
    one-dimensional.  However, there is no restriction nor guarantee on the
    type and size of its elements.

DataDescriptor:
    A wrapper or reference to concrete data in memory, on disk, or elsewhere
    that is directly accessible by the low-level Blaze runtime.  Consists of
    buffers and streams (or lists of buffers or of streams).  Buffers are like
    Numpy arrays or Python memoryviews, and can potentially be strided views
    of contiguous memory.  Streams are like character devices in the Unix
    model, with a few additional parameters for optimal chunked reading.

    DataDescriptors themselves are not Indexable; they should be thought of as
    a glorified "pointer" to bulk data, or a handle for a stream.  They do
    have the concept of a traditional Numpy dtype to specify the kinds of
    elements they return, but they do not need to implement any slicing,
    broadcasting, or shape-related functionality.
    
    Internally, Blaze tries to use DataDescriptors as much as possible,
    instead of shipping memory around.  Ultimately, kernel functions and other
    low-level parts of the runtime will use DataDescriptors to actually read
    or copy data, and they use the metadata and flags on DataDescriptors to
    adapt their inner loops to the layout of the data.

ByteProvider:
    An Indexable object which only returns DataDescriptors.  This type of
    Indexable object is specifically distinguished from other Indexable
    objects because Blaze arrays/tables are recursive data structures, and can
    nest freely.  A ByteProvider is a primitive indexed object that is
    guaranteed to produce concrete data.  It does not have a concept of
    dimensions or shape.

DataSpace:
    The most fundamental and most general multi-dimensional object in Blaze.
    It has a "shape", although it should be noted that the Blaze Shape object
    is a very rich structure, beyond the standard "list of strides and
    offsets" which exists in Numpy and other dense, matrix-oriented libraries.

    The methods and operations available on DataSpace are related to
    selection, query, and reshape of the coordinate structure defined by the
    space's Shape.  There are no general arithmetic functions (as there are
    with Arrays) and there are no indexing and set-oriented operations (as
    there are with Tables).
    
    DataSpace is Indexable, and so can be generically treated as a 1D array of
    its outermost dimension, but there are no restrictions on the shape or
    type of each element in it.  In the most general case, it would need to be
    treated as a Stream of arbitrarily-shaped subspaces.

Shape:
    A generalization of the "tuple of integral sizes" approach found in
    Numpy, C, and FORTRAN.  The Shape tuple consists of a variety of Primitive
    and Complex dimension specifiers, including Fixed integers (corresponding
    to Numpy fixed-length dimensions), Variable-length dimensions, Union,
    Enum, Bitfield, and CType (for traditional C data types).  Most
    significantly, there is also a Record shape type, which is composed of
    additional specifiers for its named fields.  It is the Record Shape which
    gives rise to the distinction between Blaze's Tabular and ArrayLike
    objects.

Tabular:
    Base interface for table-like objects in Blaze.  Includes methods and
    operators which are useful for tabular views of data.  For a DataSpace
    to be Tabular, its Shape must embed a Record as its innermost dimension.
    (Records can embed other Records.)

    In general, all of the aggregation, indexing, and relational operations
    depend on the dimensions and measures established by the Record in the
    a dataspace's Shape.  All of the vectorized mathematical operations that
    one would expect are also available on Tabular objects, but it is possible
    that some operations have different default behaviors than for ArrayLike
    objects (especially with regards to NaNs and missing data).

NDTable:
    A Tabular object with lazy evaluation of data, but immediate evaluation
    of index space transformations.  Once data has been evaluated, then it
    is cached on the NDTable object.  The NDTable is intended for library
    and infrastructure code, and its interface is designed with consistency
    and programmatic handling of errors as the primary focus, and command-line
    usability as secondary.

Table:
    A node in an expression graph on Tabular objects; basically an even more
    lazily evaluated NDTable.  Even index transformations are not computed
    immediately, but rather on-demand.  The Table is meant to be used at 
    a REPL and in simple scripts, and its interface and methods are designed
    for usability by end-users.


================
Objects In Depth
================

DataSpace
=========

DataSpace, combined with DataShape, can also be thought of as recursive or
array-oriented means for expressing a nested tree coordinate structure. 


DataShape
=========

The generalized DataShape describes a hierarchical (possibly heterogenous)
tree of coordinates as a tuple of specifications of some recursive function.
In general, this is an impossible task to do and still preserve any useful
metadata that can serve the needs of a task- or data-parallelizing compiler.
However, certain types of tree structures do lend themselves to this
description, and if the heterogeneity falls within a few different types, then
it can also be parameterized.

Numpy Shapes, Two Ways
----------------------

For instance, the classical C/FORTRAN/Numpy N-D array is usually described
with a shape that is a tuple of integers.  Each integer indicates the number
of subspaces which are stacked together, and the dimensions of each subspace
is then specified by the remainder of the shape tuple.  

There is also a natural geometric interpretation of this.  If the shape tuple
has length N, then the left-most integer can be interpreted as defining a
vector in an N-dimensional space that is dual to the subspace defined by
shape[1:N]. The shape[0:N] generates the space by extruding the subspace
shape[1:N] along that vector, much like an outer product.  This outer product
formulation is recursive, and one can start with the right-most integer in a
shape tuple and build out the entire space.

In this multi-dimensional, dense cube view, transpose() and reshape() are just
volume-preserving transformations of this cube.  Since each dimension is
defined by a vector, there is no simple way to label dimensions or create any
address space that does not have a strict mapping onto the integers.  Sparse
arrays, categorical dimensions, and even hierarchical dimensions are usually
handled by enumerating a tuple space of keys.  While this makes selection of
individual elements possible, it can be awkward to slice or select ranges,
because the topological structure of the index space has been collapsed onto
the integers.

Another, somewhat unorthodox, view is to consider the coordinate space not as
a multi-dimensional dense cube, but rather as a hierarchical tree of indices.
Each tree node has an ordered set of child nodes, and every set of child nodes
is numbered starting at 0.  (The actual data is located in the terminal leaf
nodes of the tree.)  The Numpy-style selection of "planes" of subspaces, e.g.
a[:, 2, :], is akin to selecting all the child nodes of a certain index number
(in this case, 2) at a specific layer or level of the tree from the root.

This tree-view of traditional dense arrays has some nice properties.  Since it
is intrinsically about enumerating a set of leaf nodes, and there is no
implicit geometric structure, it's easier to visualize how ragged
(variable-length) arrays and other irregularities might fit into the overall
scheme.  "Shape" is a property of the tree and the non-leaf nodes, and the
tree-view makes it more apparently how hierarchical and other indexing
approaches may lead to equally valid and useful mappings over the set of
values in the leaf nodes.

.. TODO: This following section may need to be updated in light of the
   concept model above.

.. =========================================================================

NDTable
=======

An NDTable is a Python object that allows N-dimensional + "field" indexing
using indexed byte-interfaces.  It is a generalization of NumPy, Pandas, data
array, larry, ctable, and CArray, and is meant to serve as a foundational
abstraction on which to build out-of-core and distributed algorithms by
focusing attention away from moving data to the code and rather layering
interpretation on top of data that exists. 

There are two interfaces to the NDTable that are critical: 

    1) Domain expert interface that allows easy construction, indexing,
       manipulation and computation with an ND table.  

    2) Algorithm writers and developers searching for a unified API that allows
       coherent communication about the structure of data for optimization. 

It is intended to be *the* glue that holds the PyData ecosystem together.   It
has an interface for domain-experts to query their information and an interface
for algorithm writers to create out-of-core algorithms against.   The long-term
goal is for NDTable to be the interface that *all* pydata projects interact
with. 

It is the calculations / operations that can be done with an NDTable that will
ultimately matter and define whether or not it gets used in the Python
ecosystem.  We need to make it easy to create these calculations and algorithms
and push to grow this ecosystem rapidly.  

At the heart of NDTable is a delayed-evaluation system (like SymPy, Theano, and
a host of other tools).  As a result, every operation actually returns a node
of a simple, directed graph of functions and arguments (also analagous to
DyND).  However, as far as possible, these nodes will propagate meta-data and
meta-compute (such as dtype, and shape, and index-iterator parameters so that
the eventual calculation framework can reason about how to manage the
calculation. 

.. Some nodes of the expression graph are "reified" NDTables (meaning they are no
   longer an expression graph but a collection of indexed byte-buffers). 

NDTable is a Generalization of NumPy
====================================

.. TODO: This section in particular is an expression of older thoughts and 
   ideas about the NDTable and the object hierarchy.

We would like the NDTable to be a Generalization of NumPy.   Whether this means
that the NDTable augments or replaces NumPy (on the Python side) in the future
has yet to be determined.   For now, it will augment NumPy and provide
compatibility whenever possible.

In addition to the ufuncs defined over NumPy arrays, NumPy defines basically 3
things that we wish to generalize
     
     * A data-type
     * A shape
     * A strides map to a single data-buffer (a linear, formula-based index)

These concepts are generalized via the concept of indexed byte-buffers.
NumPy-style arrays consist of a single data-segment that can be explained via a
linear indexing function with strides as the coefficients.   In addition, to
serving as a dispatch mechanism for functions, the dtype also participates in
indexing operations via the itemsize, and structure data-types. 

NumPy is a pointer to memory with "indexing" information provided by strides.
While we don't typically think of NumPy arrays as having an "index", the
strides attribute provides a linear, formula-based index so that A[I] maps to
mem[I \cdot s] where mem is the memory buffer for the array, I is the
index-vector, and s is the strides vector.

NDTable generalizes this notion to multiple data-buffers and (necessarily)
different kinds of indexes.    Domain-experts are not going to care (much)
about the details of how this lays out (just as today most users of NumPy don't
care about the strides).   However, algorithm writers will care a great deal
about the actual data-layout of an NDTable and want to process the elements of
an array in the easiest possible way.   There must be interfaces that allow
algorithm writers to get at this information.    

One concept which will remain true is that some algorithms will work faster and
more optimally with data laid out in a certain way.  As a result, an ndtable
may have several data-layouts to choose from which can be selected as needed
for optimization. 

So, at the core of every (reified) NDTable there is a collection of
byte-interfaces and an index (or collection of indexes) that allows mapping
calls to __getitem__ to the appropriate interface.  These byte-interfaces, how
they are indexed (including the meaning of shape), and what elements represent,
are the fundamental building-blocks of the NDTable. 

Byte-interfaces
============

Byte-interfaces are objects that connect to raw memory, disk-files, HTTP
requests, GPU memory, data-base connections, measurements, generated data, or
any other byte-streams).   Care is taken so that memory-based byte-buffers can
be as fast as possible.  

A byte-interface is either generator-like (with a next(N) method), file-like
(read, write) or memory-like (getref (N,byte-stride)).   Default caches are
used for generator-like and file-like byte-interfaces which can be over-ridden
by the object.

Index
====

An index is a mapping from a domain specification to a collection of
byte-interfaces and offsets.  

IndexedBytes
=========

Shape
====

DType
=====

NDTable
======



Random Thoughts
============

Generalizing Shape
---------------

The shape attribute is an important part of every NumPy array.  For NDTable, the shape attribute may not always be a tuple of integers.   len(a.shape) will be the number of dimensions that the array holds. but the tuple may contain other objects (functions, tuples, None, etc.) depending on the complexity of the data layout and whether or not it is infinite. 

A core concept in NDTables is the dimension / field table.   One can construct a dimension / field table for every NDTable.   This table is a logical expression of what is a queryable via mapping (__getitem__) and what is extractable via attribute lookup (or mapping on the .fields object).   It allows a logical separation between "dimensions" and "measures".   It is important to note that, unlike NumPy, the fields of an ndtable do not have to be contiguous segments.  In addition, despite the appearance of the table, dimensions can be composed of multiple "elements" (hierarchical dimensions).  

Dimensions are much more generalized in NDTable from NumPy.  Dimensions can be grouped together from fields and other dimensions.   Dimensions can be automatic (standard 0..N-1 style of NumPy, or based on labels).   All NDTables can have labeled fields and labeled dimensions and labeled axes.   Labels on dimensions are "level" dtypes in that the labels are "interned" and replaced with an integer column. 

Generalizing DataType
------------------

The data-type in Python is a Python object.   In general, however, it should be a type-object (more like ctypes --- long ago I argued the other way on the Python dev list, but now I realize I was wrong).   Dtypes are just type objects defined dynamically.   There isn't a need for a separate "array-scalar".  An array-scalar is just an instance of the dtype type-object.  

In NDTable we need a way to define all kinds of data: a Data-definition language.   Just using Python's class syntax should be enough.     However, there is the potential for confusion about which attributes and methods are "virtual" and dynamic and which are "the data description itself....".   It might make sense to define a __pod__ attribute of the metaclass so that ever instance can populate it's own attribute list and this __pod__ attriubute will be 


Generalizing UFuncs
==============

On top of this basic data-structure we create algorithms and operations:  a table-function object.   The table-function object takes a kernel which deals with memory chunks and   




