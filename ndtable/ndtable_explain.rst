
NDTable
======

The NDTable is a python-level object that allows the user to query rapidly the contents of N-dimensional arrays and tables and is a common interface for algorithm writers to write against.   At its core, the NDTable is an abstract interface which defines the notion of nodes, delayed-evaluation, and index objects so that:

    1) Domain experts have a unified n-dimensional table object to manipulate and hold their data
    2) Algorithm writers and Developers have a unified API to communicate about the structure of data

It is intended to be *the* glue that holds the pydata ecosystem together.   It has an interface for domain-experts to query their information and an interface for algorithm writers to create out-of-core algorithms against.   The long-term goal is for NDTable to be the interface that *all* pydata projects interact with. 

It is the calculations / operations that can be done with an NDTable that will ultimately matter and define whether or not it gets used in the Python ecosystem.  We need to make it easy to create these calculations and algorithms and push to grow this ecosystem rapidly.  

At the heart of NDTable is a delayed-evaluation system (like SymPy, Theano, and a host of other tools).  As a result, every operation actually returns a node of a simple, directed graph of functions and arguments (just like DyND).  However, as far as possible, these nodes will propagate meta-data and meta-compute (such as dtype, and shape, and index-iterator parameters so that the eventual calculation framework can reason about how to manage the calculation. 

Some nodes of the expression graph are "reified" NDTables (meaning they are no longer an expression graph but a collection of byte-buffers with attached index-iterators). 

NDTable is a Generalization of NumPy
=========================

We would like the NDTable to be a Generalization of NumPy.   Whether this means that the NDTable augments or replaces NumPy (on the Python side) in the future has yet to be determined.   For now, it will augment NumPy and provide compatibility whenever possible.

In addition to the ufuncs defined over NumPy arrays, NumPy defines basically 3 things that we wish to generalize
     
     * A data-type
     * A shape
     * A strides map to a single data-buffer (a linear, formula-based index)

Generalizing Strides
----------------

NumPy is a pointer to memory with "indexing" information provided by strides.  While we don't typically think of NumPy arrays as having an "index", the strides attribute provides a linear, formula-based index so that A[I] maps to mem[I \cdot s] where mem is the memory buffer for the array, I is the index-vector, and s is the strides vector.

NDTable generalizes this notion to multiple data-buffers and (necessarily) different kinds of indexes.    Domain-experts are not going to care (much) about the details of how this lays out (just as today most users of NumPy don't care about the strides).   However, algorithm writers will care a great deal about the actual data-layout of an NDTable and want to process the elements of an array in the easiest possible way.   There must be interfaces that allow algorithm writers to get at this information.    

One concept which will remain true is that some algorithms will work faster and more optimally with data laid out in a certain way.  As a result, an ndtable may have several data-layouts to choose from which can be selected as needed for optimization. 

So, at the core of every (reified) ndtable there is a collection of byte-buffers and an index (or collection of indexes) that allows mapping calls to __getitem__ to the appropriate segments.  These byte-buffers, how they are indexed, and what they represent are the fundamental building-blocks of the NDTable. 

In the general case, the byte-buffers are an interface to bytes (wrapping memory, a disk-file, HTTP request, data-base connection, a measurement, generated data, or other byte-stream).   For memory, there is a fast-path (i.e. the raw pointer might be stored rather than through an interface).   The byte-interface is defined formally by a POD data-structure and then explicit C-ABI function pointers but we will start with just Python objects.

A byte-buffer interface is either generator (next(N)) or random-access (read, seek, write).  


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




