========
Overview
========

Blaze is a Generalization of NumPy
----------------------------------

.. image:: svg/numpy_plus.png
    :align: center

We would like Blaze to be a generalization of NumPy.  Whether this means
that the Array and Table objects replace NumPy arrays in the future
has yet to be determined.  For now, it will augment NumPy and provide
interoperability whenever possible.

Datashape
~~~~~~~~~

The type system in Blaze is called Datashape, and generalizes the
combination of shape and dtype in NumPy. The datashape of an array
consists of a number of dimensions, followed by an element type.

Data Descriptor
~~~~~~~~~~~~~~~

The data descriptor is the interface which exposes multi-dimensional
data to Blaze. It provides data layout, iteration and indexing
of data, no mathematical operations. This is similar to the Python
memoryview object, but with a more general multi-dimensional structure.
The data descriptor is for interfacing to data in any of local memory,
out of core storage, or distributed storage.

Array
~~~~~

The array object wraps a data descriptor, adding arithmetic, field
access, and other properties to be the main array programming object
in Blaze. Most algorithms will be written in terms of arrays.

Table
~~~~~

The table object wraps a number of named data descriptors, representing
an array of records with a field-oriented storage. It provides data-oriented
access via various queries and selections.

Blaze Functions
~~~~~~~~~~~~~~~

Blaze functions provide a way to define functions which operate on
elements or subarrays of blaze arrays. Evaluation is deferred until
the blaze.eval function is called on the result.

Blaze functions provide multiple dispatch into a table of kernels,
which may be parameterized and use runtime code generation.

