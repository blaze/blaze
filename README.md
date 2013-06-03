
Blaze 0.1
============

Pre-requisites:
  * llvmpy >= 0.11.1
  * cython >= 0.16
  * numpy >= 1.5

Install all pre-requisites and then run:

python setup.py install

Docs are generated using sphinx in the docs directory.


Overview of early functionality
===============================

Concrete Array object
---------------------

* Make immediate Array() to work with the basic data types: bools,
integers, floats, complex, strings and compound types. Variable length
should be supported too (although they won't be very efficient for now).

* Basic constructors (ones, zeros) should work decently well. The
Array.fromiter() should work too (currently it has some issues).

* Multidimensional indexing should be tested and work by returning a
NumPy array buffer initially. In the future this operation should return
a Blaze object (Array or Scalar).

* The Array() should work both in memory and on disk, and should support
appends efficiently. In particular, when it is on disk, the sequence
create-append-close-open-append-close should work well.

Operations on concrete Array objects
------------------------------------

* Implement basic binary operations: +, -, *, /

* Implement some unary operations: abs(), neg(), sin(), cos()...

* Implement an interesting operation: madd(a, b, c) -> a + b * c

All the operations above will have temporaries in memory, except for
`madd()`, which will use Blir internally. Deferred machiner will enter
into action in forthcoming releases.

Also, we will direct users to have a glimpse at the deferred
array operations example that Oscar and me setup recently in
`samples/chunked_dot.py` that uses Blir/numexpr/!NumPy.

Blaze Execution Engine
----------------------

* Setup a !ByteProvider class to uniformize the access to different
container backends: BLZ, dynd, !NumPy

* Build a very basic `execution engine` that would be able to ask for
slices to the !ByteProvider and perform the operations for them. This
will be based on Oscar's implementation for the `samples/chunked_dot.py`
example.

* The execution engine will be able to use plugable on-the-flight compilers: Blir, Numba, numexpr.
