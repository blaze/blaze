
NDTable
======

An NDTable is a python-object that allows N-dimensional + "field" indexing using indexed byte-interfaces.    It is a generalization of NumPy, Pandas, data array, larry, ctable, and CArray, and is meant to serve as a foundational abstraction on which to build out-of-core and distributed algorithms by focusing attention away from moving data to the object and rather layering interpretation on top of data that exists. 

There are two interfaces to the NDTable that are critical: 

    1) Domain expert interface that allows easy construction, indexing, manipulation and computation with an ND table.
    2) Algorithm writers and developers searching for a unified API that allows coherent communication about the structure of data for optimization. 

It is intended to be *the* glue that holds the pydata ecosystem together.   It has an interface for domain-experts to query their information and an interface for algorithm writers to create out-of-core algorithms against.   The long-term goal is for NDTable to be the interface that *all* pydata projects interact with. 

It is the calculations / operations that can be done with an NDTable that will ultimately matter and define whether or not it gets used in the Python ecosystem.  We need to make it easy to create these calculations and algorithms and push to grow this ecosystem rapidly.  

At the heart of NDTable is a delayed-evaluation system (like SymPy, Theano, and a host of other tools).  As a result, every operation actually returns a node of a simple, directed graph of functions and arguments (also analagous to DyND).  However, as far as possible, these nodes will propagate meta-data and meta-compute (such as dtype, and shape, and index-iterator parameters so that the eventual calculation framework can reason about how to manage the calculation. 

Some nodes of the expression graph are "reified" NDTables (meaning they are no longer an expression graph but a collection of indexed byte-buffers). 

NDTable is a Generalization of NumPy
=========================

We would like the NDTable to be a Generalization of NumPy.   Whether this means that the NDTable augments or replaces NumPy (on the Python side) in the future has yet to be determined.   For now, it will augment NumPy and provide compatibility whenever possible.

In addition to the ufuncs defined over NumPy arrays, NumPy defines basically 3 things that we wish to generalize
     
     * A data-type
     * A shape
     * A strides map to a single data-buffer (a linear, formula-based index)

These concepts are generalized via the concept of indexed byte-buffers.   NumPy-style arrays consist of a single data-segment that can be explained via a linear indexing function with strides as the coefficients.   In addition, to serving as a dispatch mechanism for functions, the dtype also participates in indexing operations via the itemsize, and structure data-types. 

NumPy is a pointer to memory with "indexing" information provided by strides.  While we don't typically think of NumPy arrays as having an "index", the strides attribute provides a linear, formula-based index so that A[I] maps to mem[I \cdot s] where mem is the memory buffer for the array, I is the index-vector, and s is the strides vector.

NDTable generalizes this notion to multiple data-buffers and (necessarily) different kinds of indexes.    Domain-experts are not going to care (much) about the details of how this lays out (just as today most users of NumPy don't care about the strides).   However, algorithm writers will care a great deal about the actual data-layout of an NDTable and want to process the elements of an array in the easiest possible way.   There must be interfaces that allow algorithm writers to get at this information.    

One concept which will remain true is that some algorithms will work faster and more optimally with data laid out in a certain way.  As a result, an ndtable may have several data-layouts to choose from which can be selected as needed for optimization. 

So, at the core of every (reified) ndtable there is a collection of byte-interfaces and an index (or collection of indexes) that allows mapping calls to __getitem__ to the appropriate interface.  These byte-interfaces, how they are indexed (including the meaning of shape), and what elements represent, are the fundamental building-blocks of the NDTable. 

Byte-interfaces
============

Byte-interfaces are objects that connect to raw memory, disk-files, HTTP requests, GPU memory, data-base connections, measurements, generated data, or any other byte-streams).   Care is taken so that memory-based byte-buffers can be as fast as possible.  

A byte-interface is either generator-like (with a next(N) method), file-like (read, write) or memory-like (getref (N,byte-stride)).   Default caches are used for generator-like and file-like byte-interfaces which can be over-ridden by the object.

Index
====

An index is a mapping from a domain specification to a collection of byte-interfaces and offsets.  

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




